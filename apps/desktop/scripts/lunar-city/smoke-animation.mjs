import { chromium } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const url = process.argv[2] ?? 'http://127.0.0.1:5178/lunar-city-review.html'
const output = resolve(process.argv[3] ?? '/private/tmp/lunar-city-animation-evidence')
const subject = process.argv[4] ?? 'cat'
const worker = subject.startsWith('worker:')
const leader = worker ? subject.slice(7) : subject
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
const errors = []
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(url)
  await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true', null, { timeout: 60_000 })
  await page.click('#motion')
  await page.selectOption('#leaders', `lunar-city:${worker ? 'review-worker' : 'leader'}:${leader}`)
  const states = await page
    .locator('#animation option')
    .evaluateAll(nodes => nodes.map(node => node.value).filter(Boolean))
  if (!states.includes('walk') || !states.includes('idle'))
    throw new Error('Imported locomotion and idle clips are missing')
  const samples = []
  for (const state of states.filter(value => value !== 'idle')) {
    console.log('Checking', subject, state)
    await page.selectOption('#animation', 'idle')
    await page.selectOption('#animation', state)
    const sample = await page.evaluate(
      async ({ leader, state, worker }) => {
        const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
        const { Engine } = await loadBabylonModules()
        const scene = Engine.Instances[0].scenes[0]
        const owned = value =>
          value.targetedAnimations.some(
            ({ target }) => target.metadata?.lunarCity?.[worker ? 'workerReviewId' : 'leaderId'] === leader
          )
        const group = scene.animationGroups.find(
          value => value.name === (worker ? state : `leader:${leader}:${state}`) && owned(value)
        )
        const playing = scene.animationGroups.filter(value => owned(value) && value.isPlaying).map(value => value.name)
        if (group.hasMotion !== false && (playing.length !== 1 || playing[0] !== group.name))
          throw new Error(`Clip overlap: ${playing}`)
        const mesh = scene.meshes.find(
          value => value.skeleton && value.metadata?.lunarCity?.[worker ? 'workerReviewId' : 'leaderId'] === leader
        )
        if (!mesh) throw new Error('No skinned leader mesh loaded')
        if (group.hasMotion === false) {
          if (playing.length) throw new Error('Static pose is wasting animation frames')
          return {
            state,
            clip: group.name,
            bones: mesh.skeleton.bones.length,
            targets: group.targetedAnimations.length,
            heldPose: true,
            maximumVertexDisplacementMetres: 0
          }
        }
        group.goToFrame(group.from)
        const initialFrame = group.from
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error(`No animation frame advance for ${state}`)), 10000)
          const observer = scene.onAfterRenderObservable.add(() => {
            if (
              (group.animatables[0]?.masterFrame ?? (!group.isPlaying ? group.to : initialFrame)) >
              initialFrame + 1
            ) {
              clearTimeout(timeout)
              scene.onAfterRenderObservable.remove(observer)
              resolve()
            }
          })
        })
        const advancedFrame = group.animatables[0]?.masterFrame ?? group.to
        if (!group.isPlaying) group.start(false)
        group.pause()
        group.goToFrame(group.from)
        scene.render()
        const base = Array.from(mesh.getPositionData(true, true))
        let maximum = 0
        for (const fraction of [1 / 12, 1 / 4, 1 / 2]) {
          group.goToFrame(group.from + (group.to - group.from) * fraction)
          scene.render()
          const posed = mesh.getPositionData(true, true)
          for (let i = 0; i < base.length; i += 3)
            maximum = Math.max(
              maximum,
              Math.hypot(posed[i] - base[i], posed[i + 1] - base[i + 1], posed[i + 2] - base[i + 2])
            )
        }
        const bonePose = mesh.skeleton.bones.map(bone => ({
          name: bone.name,
          matrix: Array.from(bone.getLocalMatrix().m)
        }))
        return {
          state,
          clip: group.name,
          bones: mesh.skeleton.bones.length,
          targets: group.targetedAnimations.length,
          initialFrame,
          advancedFrame,
          maximumVertexDisplacementMetres: maximum,
          bonePose
        }
      },
      { leader, state, worker }
    )
    if (!sample.heldPose && !(sample.maximumVertexDisplacementMetres > 0.001))
      throw new Error(`No real skin deformation for ${state}`)
    samples.push(sample)
    if (['walk', 'talking', 'work', 'tool-use'].includes(state))
      await page.screenshot({ path: resolve(output, `${subject.replace(':', '-')}-${state}.png`) })
  }
  await page.selectOption('#animation', 'idle')
  const final = await page.evaluate(async () => (await import('/src/app/lunar-city/review/main.ts')).reviewState())
  if (final.metrics.activeAnimations !== 0) throw new Error('Idle did not park the leader animation')
  await page.click('#motion')
  await page.waitForFunction(
    async () => (await import('/src/app/lunar-city/review/main.ts')).reviewState().metrics.activeAnimations > 0
  )
  const normalIdle = await page.evaluate(
    async ({ leader, worker }) => {
      const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
      const { Engine } = await loadBabylonModules()
      const group = Engine.Instances[0].scenes[0].animationGroups.find(
        value =>
          value.name === (worker ? 'idle' : `leader:${leader}:idle`) &&
          value.targetedAnimations.some(
            ({ target }) => target.metadata?.lunarCity?.[worker ? 'workerReviewId' : 'leaderId'] === leader
          )
      )
      return { clip: group?.name, playing: group?.isPlaying, looping: group?.loopAnimation }
    },
    { leader, worker }
  )
  if (!normalIdle.playing || !normalIdle.looping) throw new Error('Normal idle did not play its real loop')
  await page.click('#motion')
  if (errors.length) throw new Error(errors.join('\n'))
  const receipt = {
    url,
    capturedAt: new Date().toISOString(),
    character: subject,
    evidence: 'Actual WebGL skin deformation and UI clip transitions; idle parks at its initial pose',
    availableStates: states,
    samples,
    normalIdle,
    finalActiveAnimations: final.metrics.activeAnimations,
    pageErrors: errors
  }
  await writeFile(
    resolve(output, `${subject.replace(':', '-')}-animation.json`),
    JSON.stringify(receipt, null, 2) + '\n'
  )
  console.log(
    JSON.stringify(
      { leader, states, samples: samples.map(({ bonePose, ...sample }) => sample), pageErrors: errors },
      null,
      2
    )
  )
} finally {
  await browser.close()
}
