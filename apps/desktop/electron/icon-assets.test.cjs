const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const desktopRoot = path.join(__dirname, '..')
const repoRoot = path.join(desktopRoot, '..', '..')

function readPngSize(file) {
  const buffer = fs.readFileSync(file)
  assert.equal(buffer.toString('hex', 0, 8), '89504e470d0a1a0a', `${file} is not a PNG`)
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20)
  }
}

test('desktop macOS bundle icon is a real icns file', () => {
  const icon = fs.readFileSync(path.join(desktopRoot, 'assets', 'icon.icns'))
  assert.equal(icon.toString('ascii', 0, 4), 'icns')
})

test('bootstrap installer PNG icons use their declared dimensions', () => {
  const iconRoot = path.join(repoRoot, 'apps', 'bootstrap-installer', 'src-tauri', 'icons')
  assert.deepEqual(readPngSize(path.join(iconRoot, '32x32.png')), { width: 32, height: 32 })
  assert.deepEqual(readPngSize(path.join(iconRoot, '128x128.png')), { width: 128, height: 128 })
  assert.deepEqual(readPngSize(path.join(iconRoot, '128x128@2x.png')), { width: 256, height: 256 })
})

test('bootstrap installer macOS icon is a real icns file', () => {
  const icon = fs.readFileSync(
    path.join(repoRoot, 'apps', 'bootstrap-installer', 'src-tauri', 'icons', 'icon.icns')
  )
  assert.equal(icon.toString('ascii', 0, 4), 'icns')
})

test('desktop and bootstrap installer bundle icons stay in sync', () => {
  const iconRoot = path.join(repoRoot, 'apps', 'bootstrap-installer', 'src-tauri', 'icons')
  assert.deepEqual(
    fs.readFileSync(path.join(desktopRoot, 'assets', 'icon.icns')),
    fs.readFileSync(path.join(iconRoot, 'icon.icns'))
  )
  assert.deepEqual(
    fs.readFileSync(path.join(desktopRoot, 'assets', 'icon.ico')),
    fs.readFileSync(path.join(iconRoot, 'icon.ico'))
  )
})
