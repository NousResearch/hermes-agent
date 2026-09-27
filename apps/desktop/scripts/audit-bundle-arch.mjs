// audit-bundle-arch.mjs: prove every native binary inside a packed desktop
// bundle was built for the target architecture.
//
// Why: wrong-arch binaries run FINE on the build runner (Windows-on-ARM and
// Rosetta emulate x64 silently) and only fail — or quietly burn cycles in
// emulation — on user machines. One x64 pip launcher shim inside an arm64
// payload shipped exactly this way. The build must fail loudly instead.
//
// The scan walks every file, sniffs the magic bytes (PE / ELF / Mach-O,
// including fat headers), and reports any binary whose architecture does
// not match the target. Non-binaries are skipped by content, not by file
// extension — a mis-extensioned .exe still gets caught.
//
// CLI:
//   node audit-bundle-arch.mjs --arch=arm64 --root=release
//
// --root is scanned for the unpacked app dirs electron-builder produces
// (win-unpacked, win-arm64-unpacked, linux-unpacked, mac/, mac-arm64/, …).
// Auditing the unpacked tree covers the exact bytes the installer embeds:
// MSIX/DMG/AppImage add compression around it, not content.

import fs from "node:fs"
import path from "node:path"

import { isMain } from "./utils.mjs"

// ─── pure classification (unit-tested, no I/O) ─────────────────────────────

const PE_MACHINES = {
  0x014c: "ia32",
  0x01c0: "arm",
  0x01c4: "arm",
  0x8664: "x64",
  0xaa64: "arm64",
  0xa641: "arm64ec", // ARM64EC: arm64-ABI code, runs only on arm64 Windows
}

const ELF_MACHINES = {
  0x03: "ia32",
  0x28: "arm",
  0x3e: "x64",
  0xb7: "arm64",
}

const MACHO_CPUTYPES = {
  0x01000007: "x64",
  0x0100000c: "arm64",
  0x00000007: "ia32",
  0x0000000c: "arm",
}

/**
 * Classify a buffer holding the head of a file (>= 64 bytes is plenty for
 * every format except PE, whose COFF header lives at an offset named at
 * 0x3c — pass headerAt for that case; classifyFile below handles it).
 *
 * Returns null for non-binaries, or { format, arches } where arches lists
 * every architecture the file carries (>1 only for Mach-O fat binaries).
 */
export function classifyHeader(buf) {
  if (buf.length >= 0x40 && buf[0] === 0x4d && buf[1] === 0x5a) {
    // PE: the real machine field needs a second read at e_lfanew; caller
    // resolves it. Report the format and the offset to read.
    return { format: "pe", peHeaderOffset: buf.readUInt32LE(0x3c) }
  }
  if (buf.length >= 20 && buf[0] === 0x7f && buf[1] === 0x45 && buf[2] === 0x4c && buf[3] === 0x46) {
    const machine = buf.readUInt16LE(18)
    return { format: "elf", arches: [ELF_MACHINES[machine] || `unknown(0x${machine.toString(16)})`] }
  }
  if (buf.length >= 8) {
    const be = buf.readUInt32BE(0)
    // Thin Mach-O, either endianness.
    if (be === 0xfeedface || be === 0xfeedfacf) {
      const cputype = buf.readUInt32BE(4)
      return { format: "macho", arches: [MACHO_CPUTYPES[cputype] || `unknown(0x${cputype.toString(16)})`] }
    }
    if (be === 0xcefaedfe || be === 0xcffaedfe) {
      const cputype = buf.readUInt32LE(4)
      return { format: "macho", arches: [MACHO_CPUTYPES[cputype] || `unknown(0x${cputype.toString(16)})`] }
    }
    // Fat/universal: big-endian header, one arch entry per slice.
    if (be === 0xcafebabe) {
      const count = buf.readUInt32BE(4)
      // A Java .class file shares the magic; its "count" here is the
      // bytecode version (>= 45). Real fat binaries carry a few slices.
      if (count > 0 && count < 30) {
        const arches = []
        for (let i = 0; i < count && 8 + i * 20 + 4 <= buf.length; i++) {
          const cputype = buf.readUInt32BE(8 + i * 20)
          arches.push(MACHO_CPUTYPES[cputype] || `unknown(0x${cputype.toString(16)})`)
        }
        return { format: "macho-fat", arches }
      }
    }
  }
  return null
}

/**
 * Resolve a PE machine code (read at peHeaderOffset+4) to an arch name.
 * Split from classifyHeader so the walker can do the second read lazily.
 */
export function peArch(machineCode) {
  return PE_MACHINES[machineCode] || `unknown(0x${machineCode.toString(16)})`
}

/**
 * Does a classified binary satisfy the target arch?
 *  - exact match, always
 *  - macOS fat binary: any slice matching is enough (universal ships both)
 *  - arm64 targets accept arm64ec PEs (arm64-ABI by definition)
 * Everything else — including "unknown(...)" — is a mismatch: an
 * unclassifiable machine code in a shipped binary deserves a loud failure,
 * not a shrug.
 */
export function archMatches(arches, targetArch) {
  return arches.some((a) => a === targetArch || a === "anycpu" || (targetArch === "arm64" && a === "arm64ec"))
}

const COMIMAGE_FLAGS_ILONLY = 0x1
const COMIMAGE_FLAGS_32BITREQUIRED = 0x2
const COMIMAGE_FLAGS_32BITPREFERRED = 0x20000

/**
 * A .NET IL-only AnyCPU assembly carries machine 0x14c (i386) in its COFF
 * header, yet the CLR JITs it to the host's native arch (x64, arm64). Its
 * real arch lives in the CLR header flags. `readAt(offset, length)` returns
 * the file bytes there; returns true only for IL-only, non-32-bit PEs.
 */
export function isAnyCpuAssembly(readAt, peOffset) {
  const coff = readAt(peOffset, 24)
  if (coff.length < 24) return false
  const sections = coff.readUInt16LE(6)
  const optionalSize = coff.readUInt16LE(20)
  const optional = readAt(peOffset + 24, optionalSize)
  if (optional.length < optionalSize || optionalSize < 2) return false
  const dataDirs = { 0x10b: 96, 0x20b: 112 }[optional.readUInt16LE(0)]
  if (dataDirs === undefined || optionalSize < dataDirs + 15 * 8) return false
  const clrRva = optional.readUInt32LE(dataDirs + 14 * 8)
  if (!clrRva) return false
  const table = readAt(peOffset + 24 + optionalSize, sections * 40)
  for (let base = 0; base + 40 <= table.length; base += 40) {
    const virtualSize = table.readUInt32LE(base + 8)
    const virtualAddress = table.readUInt32LE(base + 12)
    const rawSize = table.readUInt32LE(base + 16)
    const rawPointer = table.readUInt32LE(base + 20)
    if (clrRva >= virtualAddress && clrRva < virtualAddress + Math.max(virtualSize, rawSize)) {
      const cor = readAt(rawPointer + clrRva - virtualAddress, 20)
      if (cor.length < 20) return false
      const flags = cor.readUInt32LE(16)
      return Boolean(flags & COMIMAGE_FLAGS_ILONLY)
        && !(flags & COMIMAGE_FLAGS_32BITREQUIRED) && !(flags & COMIMAGE_FLAGS_32BITPREFERRED)
    }
  }
  return false
}

/**
 * The unpacked-app directories electron-builder leaves under release/ for
 * a target arch. Names differ per platform: Windows/Linux use
 * <platform>[-<arch>]-unpacked with the HOST-default arch unsuffixed; mac
 * uses mac[-<arch>]/. Match by shape, not an exhaustive list, so a rename
 * in electron-builder fails the audit (no dirs found) instead of silently
 * auditing nothing.
 */
export function findUnpackedDirs(entries) {
  return entries.filter((name) => /-unpacked$/.test(name) || /^mac(-[a-z0-9]+)?$/.test(name))
}

// ─── impure walker + CLI ────────────────────────────────────────────────────

function classifyFile(filePath) {
  const fd = fs.openSync(filePath, "r")
  try {
    const head = Buffer.alloc(4096)
    const n = fs.readSync(fd, head, 0, head.length, 0)
    const sniffed = classifyHeader(head.subarray(0, n))
    if (!sniffed) return null
    if (sniffed.format !== "pe") return sniffed
    // PE: read the 6 bytes at e_lfanew — "PE\0\0" + machine.
    const peHead = Buffer.alloc(6)
    const m = fs.readSync(fd, peHead, 0, 6, sniffed.peHeaderOffset)
    if (m < 6 || peHead.readUInt32LE(0) !== 0x00004550) {
      return null // MZ without a PE header: DOS-era stub or corrupt — not a shippable binary format we know.
    }
    const readAt = (offset, length) => {
      const out = Buffer.alloc(length)
      return out.subarray(0, fs.readSync(fd, out, 0, length, offset))
    }
    const arch = peArch(peHead.readUInt16LE(4))
    if (arch === "ia32" && isAnyCpuAssembly(readAt, sniffed.peHeaderOffset)) return { format: "pe", arches: ["anycpu"] }
    return { format: "pe", arches: [arch] }
  } finally {
    fs.closeSync(fd)
  }
}

function* walkFiles(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isSymbolicLink()) continue // targets are scanned as files where they live
    if (entry.isDirectory()) {
      yield* walkFiles(full)
    } else if (entry.isFile()) {
      yield full
    }
  }
}

/**
 * Paths allowed to carry foreign-arch binaries. pip and setuptools ship
 * Windows launcher STUB TEMPLATES as package data (distlib t32/t64/w32/
 * w64.exe, setuptools cli*.exe / gui*.exe). They are not executed from
 * here; pip copies one to build a console-script shim at install time.
 * They exist on every platform and are always x86 PEs — inside an arm64
 * or linux payload they are inert bytes, not a wrong-arch bug.
 *
 * Anchored at agent-payload (not the tree root: mac nests the payload
 * under Hermes.app/Contents/Resources/). The lib segment is `Lib` on
 * Windows and `lib/python3.11` elsewhere; both separators appear in
 * relative paths depending on the build host.
 *
 * PortableGit carries .NET AnyCPU/MSIL assemblies (Git Credential Manager:
 * Avalonia.*, Atlassian.*, Microsoft.*, System.*, etc.) across mingw64/bin,
 * mingw64/lib, and mingw64/libexec/git-core. Their PE machine field is 0x14c
 * (ia32) because .NET assemblies are format-neutral — the CLR JITs them to
 * the native arch at load time. It also ships usr/libexec/getprocaddr32.exe,
 * a 32-bit MSYS2 helper. The staging script already PE-probes cmd/git.exe
 * itself; the bundle audit does not need to re-audit PortableGit's internal
 * MSYS2/.NET layout.
 */
const EXEMPT_PATTERNS = [
  // pip / setuptools launcher STUB TEMPLATES. x86 PE templates pip copies
  // when it writes a console-script shim. Present on every platform.
  // Live under the payload python store entry AND the relocatable venv.
  /agent-payload[/\\](tools[/\\]python-[^/\\]+[/\\](Lib|lib[/\\]python[\d.]+)|venv[/\\](Lib|lib[/\\]python[\d.]+))[/\\]site-packages[/\\](setuptools|pip[/\\]_vendor[/\\]distlib)[/\\]/i,
  // PortableGit internals under the payload tool store.
  /agent-payload[/\\]tools[/\\]git(-[^/\\]+)?[/\\](mingw64|clangarm64|usr|cmd)[/\\]/i,
  // discord.py ships x64 and x86 opus; opus.py loads by bitness.
  /agent-payload[/\\]venv[/\\](Lib|lib[/\\]python[\d.]+)[/\\]site-packages[/\\]discord[/\\]bin[/\\]libopus-0\.(x64|x86)\.dll$/i,
  // pvporcupine ships one native lib per OS/arch in the same wheel.
  // Only the matching sibling is loaded.
  /agent-payload[/\\]venv[/\\](Lib|lib[/\\]python[\d.]+)[/\\]site-packages[/\\]pvporcupine[/\\]lib[/\\]/i,
  // debugpy ships attach helpers for every OS/arch in one wheel.
  /agent-payload[/\\]venv[/\\](Lib|lib[/\\]python[\d.]+)[/\\]site-packages[/\\]debugpy[/\\]_vendored[/\\]pydevd[/\\]/i,
  // agent-browser ships no native win-arm64 build; on win32-arm64 the
  // payload stages the x64 exe, which Windows runs under built-in
  // emulation (its own postinstall falls back to x64 on arm64).
  /agent-payload[/\\]tools[/\\]agent-browser-[^/\\]+[/\\]bin[/\\]agent-browser-win32-x64\.exe$/i,
  // CfT publishes no native win-arm64 Chromium, so PM stages the win64
  // (x64) build and Windows runs it under emulation. The zip extracts
  // into chrome-win64 inside the chromium-<rev> store entry. Scoping to
  // that segment keeps linux/darwin Chromium audited.
  /agent-payload[/\\]tools[/\\]chromium-[^/\\]+[/\\]chrome-win64[/\\]/i,
  // The uv wheel/build cache (uv-cache/) is DELIBERATELY shipped with the
  // payload for warm rebuilds of the mutable venv (pm bundle copies it).
  // It holds cached sdists/archives that uv may have built for ANY arch
  // (x64/ia32 PEs on an arm64 payload) — inert cache bytes, never loaded
  // at runtime. Same class as the fetch-* prune: dead weight, not a
  // binary. The audit must not fail a payload for shipping its own cache.
  /agent-payload[/\\]uv-cache[/\\]/i,
]

export function isExemptPath(relPath) {
  return EXEMPT_PATTERNS.some((p) => p.test(relPath))
}

export function auditTree(rootDir, targetArch) {
  const mismatches = []
  let binaries = 0
  let exempted = 0
  for (const file of walkFiles(rootDir)) {
    let info
    try {
      info = classifyFile(file)
    } catch {
      continue // unreadable file: electron-builder would have failed on it already
    }
    if (!info) continue
    binaries += 1
    if (!archMatches(info.arches, targetArch)) {
      const rel = path.relative(rootDir, file)
      if (isExemptPath(rel)) {
        exempted += 1
        continue
      }
      mismatches.push({ file: rel, format: info.format, arches: info.arches })
    }
  }
  return { binaries, exempted, mismatches }
}

function main() {
  const args = Object.fromEntries(
    process.argv.slice(2).map((a) => {
      const [k, ...v] = a.replace(/^--/, "").split("=")
      return [k, v.join("=")]
    })
  )
  const targetArch = args.arch
  const root = args.root
  if (!targetArch || !root) {
    console.error("usage: audit-bundle-arch.mjs --arch=<x64|arm64> --root=<release dir>")
    process.exit(2)
  }

  const dirs = findUnpackedDirs(fs.readdirSync(root)).map((d) => path.join(root, d))
  if (dirs.length === 0) {
    console.error(`audit-bundle-arch: no unpacked app directory under ${root} — nothing was audited, failing`)
    process.exit(1)
  }

  let failed = false
  for (const dir of dirs) {
    const { binaries, exempted, mismatches } = auditTree(dir, targetArch)
    if (binaries === 0) {
      console.error(`audit-bundle-arch: ${dir}: found no native binaries at all — the scan is broken, failing`)
      failed = true
      continue
    }
    if (mismatches.length > 0) {
      failed = true
      console.error(`audit-bundle-arch: ${dir}: ${mismatches.length} binaries do not match --arch=${targetArch}:`)
      for (const m of mismatches) {
        console.error(`  [${m.format} ${m.arches.join("+")}] ${m.file}`)
      }
    } else {
      const exemptNote = exempted > 0 ? ` (${exempted} exempt launcher stubs)` : ""
      console.log(`audit-bundle-arch: ${dir}: ${binaries} native binaries, all ${targetArch}${exemptNote}`)
    }
  }
  process.exit(failed ? 1 : 0)
}

if (isMain(import.meta.url)) {
  main()
}
