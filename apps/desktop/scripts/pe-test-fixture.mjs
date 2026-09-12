import fs from 'node:fs'

// Structurally complete fixture only; this is not a runnable Windows program.
export function peFixture(label = 'original-package') {
  const data = Buffer.alloc(0x400)
  data.write('MZ')
  data.writeUInt32LE(0x80, 0x3c)
  data.write('PE\0\0', 0x80)
  data.writeUInt16LE(0x8664, 0x84)
  data.writeUInt16LE(1, 0x86)
  data.writeUInt32LE(0x200, 0xa8)
  data.writeUInt32LE(0x200, 0xac)
  data.write(label, 0x200)
  return data
}

export function writeFixture(file, data, options) {
  return fs.writeFileSync(file, file.endsWith('.exe') ? peFixture(data) : data, options)
}

export function readFixture(file, options) {
  if (!file.endsWith('.exe')) return fs.readFileSync(file, options)
  return fs.readFileSync(file).subarray(0x200).toString('utf8').replace(/\0+$/, '')
}
