import { test, expect, _electron } from '@playwright/test'
import { setupMockBackend, waitForAppReady, buildAppEnv, PACKAGED_BINARY_PATH } from './fixtures'

test('packaged automation composer opens with the real isolated backend', async () => {
  test.setTimeout(120000)
  const fixture = await setupMockBackend()
  await fixture.app.close()
  const app = await _electron.launch({executablePath:PACKAGED_BINARY_PATH,args:['--disable-gpu','--no-sandbox'],env:buildAppEnv(fixture.sandbox,{HERMES_DESKTOP_HERMES_ROOT:'C:/w/hermes-desktop-test-build'})})
  try {
    const page=await app.firstWindow()
    await waitForAppReady({...fixture,app,page},90000)
    await page.getByRole('button',{name:'Add files and actions',exact:true}).first().click()
    await page.getByRole('menuitem',{name:/Create automation/}).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await expect(page.getByRole('button',{name:'Heartbeat',exact:true})).toBeVisible()
    await page.screenshot({path:'../../.automation-evidence/packaged-dialog.png'})
  } finally {await app.close();await fixture.cleanup()}
})
