import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { setupMockBackend, waitForAppReady } from './fixtures'

for (const type of ['Loop', 'Heartbeat'] as const) {
  test(`${type} can be created and fires through the real idle scheduler`, async () => {
    test.setTimeout(180000)
    const fixture = await setupMockBackend()
    fixture.app.process().on('exit', (code, signal) => console.log('APP EXIT', type, code, signal))
    fixture.page.on('crash', () => console.log('RENDERER CRASH', type))
    fixture.page.on('close', () => console.log('PAGE CLOSED', type))
    try {
      await waitForAppReady(fixture, 90000)
      const page = fixture.page
      const input = page.locator('[data-slot="composer-rich-input"]').first()
      await input.fill('Hello. Local scheduler test.')
      await input.press('Enter')
      await expect(page.getByText('Hello from the mock inference server! The full boot chain is working.', { exact: true })).toBeVisible({ timeout: 30000 })
      await page.getByRole('button', {name:'Add files and actions', exact:true}).first().click()
      await page.getByRole('menuitem', {name:/Create automation/}).click()
      await page.getByRole('button', {name:type, exact:true}).click()
      const prompt = `Respond with a greeting for the ${type} automation test`
      await page.getByLabel(`${type} prompt`, {exact:true}).fill(prompt)
      await page.getByLabel('Interval', {exact:true}).fill('60')
      if(type === 'Loop') await page.getByLabel(/Run limit/).fill('1')
      await page.screenshot({path:`../../.automation-evidence/${type.toLowerCase()}-form.png`})
      await page.getByRole('button', {name:type==='Loop'?'Start loop':'Create heartbeat', exact:true}).click()
      await expect(page.getByRole('dialog')).not.toBeVisible({timeout:15000})
      await expect(page.locator('body')).toContainText(type)
      const createdAt = Date.now()
      await expect.poll(()=>fixture.mock.receivedPrompts.some(s=>s.includes(prompt)), {timeout:type==='Heartbeat'?100000:30000, intervals:[1000]}).toBe(true)
      console.log(`${type} scheduler observed after ${Date.now()-createdAt} ms`)
      await expect(page.locator('body')).not.toContainText('failed to render')
      await page.screenshot({path:`../../.automation-evidence/${type.toLowerCase()}-fired.png`})
    } catch (error) {
      const evidenceDir=path.resolve('../../.automation-evidence', `${type.toLowerCase()}-failure-${Date.now()}`)
      fs.mkdirSync(evidenceDir,{recursive:true})
      console.log('ORIGINAL FAILURE', String(error))
      try { fs.cpSync(fixture.sandbox.hermesHome,path.join(evidenceDir,'agent-home'),{recursive:true}) } catch (copyError) { console.log('Evidence copy failed', String(copyError)) }
      console.log('FAILURE SANDBOX',evidenceDir)
      console.log('AUTOMATION FAILURE UI', await fixture.page.locator('body').innerText().catch(()=>'<page unavailable>'))
      await fixture.page.screenshot({path:`../../.automation-evidence/${type.toLowerCase()}-failure.png`}).catch(()=>{})
      throw error
    } finally {await fixture.cleanup()}
  })
}
