const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const mainSource = () => fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')

test('link-title fast path does not let curl follow redirects outside URL guard validation', () => {
  assert.equal(mainSource().includes("'--location'"), false)
})

test('fetchLinkTitle does not fall back to raw BrowserWindow navigation for untrusted URLs', () => {
  const source = mainSource()
  const fetchLinkTitleBody = source.match(/function fetchLinkTitle\(rawUrl\) \{[\s\S]*?\n\}/)?.[0] || ''
  assert.equal(fetchLinkTitleBody.includes('fetchHtmlTitleWithRenderer'), false)
})
