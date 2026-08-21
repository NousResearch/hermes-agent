import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadGenerateAvatarImage() {
  const start = source.indexOf('async function generateAvatarImage(')
  const end = source.indexOf('/** Shape grid + color swatches', start)

  assert.notEqual(start, -1, 'generateAvatarImage is missing')
  assert.notEqual(end, -1, 'avatar helper boundary is missing')

  const calls = []
  const context = {
    host: {
      request: async (...args) => {
        calls.push(args)

        return { success: true, image_data: 'data:image/png;base64,avatar' }
      }
    }
  }

  vm.runInNewContext(`${source.slice(start, end)}\nglobalThis.generateAvatarImage = generateAvatarImage`, context, {
    filename: 'plugin.js'
  })

  return { calls, generateAvatarImage: context.generateAvatarImage }
}

test('avatar generation gives the image RPC a 90-second deadline', async () => {
  const { calls, generateAvatarImage } = loadGenerateAvatarImage()

  const image = await generateAvatarImage('health', 'Health', 'Private health intelligence agent')

  assert.equal(image, 'data:image/png;base64,avatar')
  assert.equal(calls.length, 1)
  assert.equal(calls[0][0], 'image.generate')
  assert.equal(calls[0][2], 90_000)
})