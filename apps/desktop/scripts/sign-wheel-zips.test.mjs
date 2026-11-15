import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { signWheelZipMembers } from './sign-wheel-zips.mjs'

function crc32(data) {
  const table = new Uint32Array(256)
  for (let i = 0; i < 256; i++) {
    let c = i
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[i] = c >>> 0
  }
  let crc = -1
  for (const byte of data) crc = (crc >>> 8) ^ table[(crc ^ byte) & 0xff]
  return (crc ^ -1) >>> 0
}

function storedZip(members) {
  const chunks = []
  const central = []
  let offset = 0
  for (const [member, data] of Object.entries(members)) {
    const nameBuf = Buffer.from(member)
    const crc = crc32(data)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt32LE(crc, 14)
    local.writeUInt32LE(data.length, 18)
    local.writeUInt32LE(data.length, 22)
    local.writeUInt16LE(nameBuf.length, 26)
    chunks.push(local, nameBuf, data)
    const entry = Buffer.alloc(46)
    entry.writeUInt32LE(0x02014b50, 0)
    entry.writeUInt16LE(20, 4)
    entry.writeUInt16LE(20, 6)
    entry.writeUInt32LE(crc, 16)
    entry.writeUInt32LE(data.length, 20)
    entry.writeUInt32LE(data.length, 24)
    entry.writeUInt16LE(nameBuf.length, 28)
    entry.writeUInt32LE(offset, 42)
    central.push(entry, nameBuf)
    offset += 30 + nameBuf.length + data.length
  }
  const centralBuf = Buffer.concat(central)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(Object.keys(members).length, 10)
  end.writeUInt32LE(centralBuf.length, 12)
  end.writeUInt32LE(offset, 16)
  return Buffer.concat([...chunks, centralBuf, end])
}

function makeWheel(payload, name, members) {
  const wheelDir = path.join(payload, 'uv-cache', 'wheels-v6', 'pypi', name)
  fs.mkdirSync(wheelDir, { recursive: true })
  const wheelPath = path.join(wheelDir, `${name}-1.0.0-py3-none-any.whl`)
  fs.writeFileSync(wheelPath, storedZip(members))
  return wheelPath
}

describe('signWheelZipMembers', () => {
  it('is a no-op without an identity', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wz-'))
    try {
      const payload = path.join(root, 'agent-payload')
      makeWheel(payload, 'pure-pkg', { 'pure/pkg.py': Buffer.from('x = 1') })
      expect(signWheelZipMembers(payload, { identity: null }))
        .toEqual({ wheels: 1, signed: 0, failed: 0 })
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('reports zero for payloads without a uv-cache', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wz-'))
    try {
      expect(signWheelZipMembers(path.join(root, 'empty'), { identity: 'ABC' }))
        .toEqual({ wheels: 0, signed: 0, failed: 0 })
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})

describe('signWheelZipMembers machinery', () => {
  it('walks Mach-O members and gates on verify', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wz-'))
    try {
      const payload = path.join(root, 'agent-payload')
      // Mach-O magic bytes: 0xfeedfacf (64-bit little endian swapped)
      const macho = Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0, 1, 2, 3])
      makeWheel(payload, 'native-pkg', {
        'native/_speedups.so': macho,
        'native/data.json': Buffer.from('{}'),
      })
      const calls = []
      // The stub simulates unzip/zip/codesign on the stored-zip format the
      // fixture writes: unzip materializes members, codesign mutates the last
      // byte (a fake embedded signature), zip repacks from disk.
      const membersOf = wheelPath => {
        const buf = fs.readFileSync(wheelPath)
        const out = {}
        let off = 0
        while (buf.readUInt32LE(off) === 0x04034b50) {
          const nameLen = buf.readUInt16LE(off + 26)
          const size = buf.readUInt32LE(off + 18)
          const name = buf.toString('utf8', off + 30, off + 30 + nameLen)
          out[name] = buf.subarray(off + 30 + nameLen, off + 30 + nameLen + size)
          off += 30 + nameLen + size
        }
        return out
      }
      let original = null
      const fakeExec = (command, args, options = {}) => {
        calls.push([command, ...args])
        if (command === 'unzip') {
          const wheel = args.find(a => a.endsWith('.whl'))
          original = membersOf(wheel)
          const dest = args[args.indexOf('-d') + 1]
          for (const [name, data] of Object.entries(original)) {
            fs.mkdirSync(path.join(dest, path.dirname(name)), { recursive: true })
            fs.writeFileSync(path.join(dest, name), data)
          }
        } else if (command === 'codesign') {
          const target = args[args.length - 1]
          const buf = fs.readFileSync(target)
          buf[buf.length - 1] ^= 0xff
          fs.writeFileSync(target, buf)
        } else if (command === 'zip') {
          const out = args[args.indexOf('-r') + 2]
          const src = options.cwd
          const members = {}
          const walk = dir => {
            for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
              const p = path.join(dir, ent.name)
              if (ent.isDirectory()) walk(p)
              else members[path.relative(src, p)] = fs.readFileSync(p)
            }
          }
          walk(src)
          fs.writeFileSync(out, storedZip(members))
        }
      }
      const result = signWheelZipMembers(payload, { identity: 'DEVID', exec: fakeExec })
      expect(result.signed).toBe(1)
      const commands = calls.map(c => c[0])
      expect(commands).toContain('unzip')
      expect(commands).toContain('codesign')
      expect(commands).toContain('zip')
      // The verify gate ran codesign --verify against the repacked extraction.
      expect(calls.filter(c => c[0] === 'codesign' && c.includes('--verify')).length).toBe(1)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('skips wheels without Mach-O members without touching them', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'wz-'))
    try {
      const payload = path.join(root, 'agent-payload')
      const wheelPath = makeWheel(payload, 'pure-pkg', { 'pure/pkg.py': Buffer.from('x = 1') })
      const before = fs.readFileSync(wheelPath)
      const calls = []
      const result = signWheelZipMembers(payload, { identity: 'DEVID', exec: (...a) => calls.push(a) })
      expect(result.signed).toBe(0)
      expect(fs.readFileSync(wheelPath).equals(before)).toBe(true)
      // Only the probe unzip runs; no codesign, no repack.
      expect(calls.filter(c => c[0] !== 'unzip').length).toBe(0)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})
