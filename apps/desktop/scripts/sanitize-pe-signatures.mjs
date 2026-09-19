/**
 * sanitize-pe-signatures.mjs — repair PE certificate tables that point past
 * the end of their own file, before anything tries to sign the tree.
 *
 * Why this exists: python-build-standalone runs `llvm-strip` over every DLL
 * in its Windows distributions. Stripping drops the trailing Authenticode
 * blob but leaves IMAGE_DIRECTORY_ENTRY_SECURITY in the optional header
 * still naming its old offset and size. The result is a PE whose certificate
 * table begins at (or past) EOF.
 *
 * signtool refuses such a file with 0x800700C1 (ERROR_BAD_EXE_FORMAT), and
 * AppxSIP inspects every PE inside an MSIX, so ONE of them fails the whole
 * package signing — which is how a payload DLL nobody imports took down both
 * Windows bundled release jobs. Upstream exempted the MSVC redists from
 * stripping (python-build-standalone#856) but not tcl/tk, and the same class
 * can arrive from any stripped third-party dist we bundle.
 *
 * The repair is to zero the dangling directory entry: eight bytes that say
 * "there is no certificate here", which is already true. The bytes the entry
 * pointed at are gone; nothing else in the file references them. A signature
 * that is actually present is never touched — a valid entry is left exactly
 * as it is, so this can never strip a real signature.
 */

import fs from "node:fs"
import path from "node:path"

import { isMain } from "./utils.mjs"

const PE_SIGNATURE = 0x00004550
const COFF_HEADER_SIZE = 24 // PE signature (4) + COFF file header (20)
const PE32_MAGIC = 0x10b
const PE32PLUS_MAGIC = 0x20b
const SECURITY_DIRECTORY_INDEX = 4
const DATA_DIRECTORY_ENTRY_SIZE = 8

/**
 * Offset of NumberOfRvaAndSizes within the optional header. PE32 ends its
 * fixed fields at 92. PE32+ drops BaseOfData (-4) and widens four fields
 * from 4 to 8 bytes (+16), so its value sits 12 bytes later, at 108.
 *
 * @param {number} magic optional header magic (0x10b PE32, 0x20b PE32+)
 * @returns {number | null} byte offset, or null when the magic is unknown
 */
export function numberOfRvaAndSizesOffset(magic) {
  if (magic === PE32PLUS_MAGIC) return 108
  if (magic === PE32_MAGIC) return 92
  return null
}

/**
 * Byte offset of the security data directory entry, relative to the start
 * of the file.
 *
 * @param {number} peHeaderOffset value of e_lfanew (offset of "PE\0\0")
 * @param {number} magic optional header magic
 * @returns {number | null} absolute file offset, or null for an unknown magic
 */
export function securityDirectoryOffset(peHeaderOffset, magic) {
  const rvaCount = numberOfRvaAndSizesOffset(magic)
  if (rvaCount === null) return null
  return (
    peHeaderOffset +
    COFF_HEADER_SIZE +
    rvaCount +
    4 + // NumberOfRvaAndSizes itself
    SECURITY_DIRECTORY_INDEX * DATA_DIRECTORY_ENTRY_SIZE
  )
}

/**
 * Decide what a security directory entry describes. For this directory
 * alone the "RVA" field is a plain file offset, not a virtual address, so
 * it is directly comparable against the file size.
 *
 * "dangling" is the repairable case and is deliberately narrow: the entry
 * must name a region that cannot exist in this file. Anything that fits
 * inside the file is a real signature and is reported as "present" so the
 * caller leaves it alone.
 *
 * @param {{ certOffset: number, certSize: number, fileSize: number }} entry
 * @returns {"absent" | "present" | "dangling"}
 */
export function classifySecurityDirectory({ certOffset, certSize, fileSize }) {
  if (certOffset === 0 && certSize === 0) return "absent"
  // A zero offset with a nonzero size (or the reverse) names no usable
  // region either; both halves must be sane for a signature to be readable.
  if (certOffset === 0 || certSize === 0) return "dangling"
  if (certOffset >= fileSize) return "dangling"
  if (certOffset + certSize > fileSize) return "dangling"
  return "present"
}

/**
 * Read the security directory of a PE file.
 *
 * @param {string} file
 * @returns {{ offsetInFile: number, certOffset: number, certSize: number, fileSize: number } | null}
 *   null when the file is not a PE with data directories at all.
 */
export function readSecurityDirectory(file) {
  const fd = fs.openSync(file, "r")
  try {
    const fileSize = fs.fstatSync(fd).size
    if (fileSize < 0x40) return null

    const mz = Buffer.alloc(0x40)
    fs.readSync(fd, mz, 0, 0x40, 0)
    if (mz[0] !== 0x4d || mz[1] !== 0x5a) return null

    const peHeaderOffset = mz.readUInt32LE(0x3c)
    if (peHeaderOffset <= 0 || peHeaderOffset + COFF_HEADER_SIZE > fileSize) return null

    const coff = Buffer.alloc(COFF_HEADER_SIZE)
    fs.readSync(fd, coff, 0, COFF_HEADER_SIZE, peHeaderOffset)
    if (coff.readUInt32LE(0) !== PE_SIGNATURE) return null

    const optionalHeaderSize = coff.readUInt16LE(20)
    if (optionalHeaderSize === 0) return null // object file: no data directories

    const magicBuf = Buffer.alloc(2)
    fs.readSync(fd, magicBuf, 0, 2, peHeaderOffset + COFF_HEADER_SIZE)
    const offsetInFile = securityDirectoryOffset(peHeaderOffset, magicBuf.readUInt16LE(0))
    if (offsetInFile === null) return null
    if (offsetInFile + DATA_DIRECTORY_ENTRY_SIZE > peHeaderOffset + COFF_HEADER_SIZE + optionalHeaderSize) {
      return null // header declares fewer directories than the security slot
    }

    const entry = Buffer.alloc(DATA_DIRECTORY_ENTRY_SIZE)
    fs.readSync(fd, entry, 0, DATA_DIRECTORY_ENTRY_SIZE, offsetInFile)
    return {
      offsetInFile,
      certOffset: entry.readUInt32LE(0),
      certSize: entry.readUInt32LE(4),
      fileSize,
    }
  } finally {
    fs.closeSync(fd)
  }
}

/**
 * Zero the eight bytes of a security directory entry in place.
 *
 * @param {string} file
 * @param {number} offsetInFile offset of the entry, from readSecurityDirectory
 * @returns {void}
 */
export function clearSecurityDirectory(file, offsetInFile) {
  const fd = fs.openSync(file, "r+")
  try {
    fs.writeSync(fd, Buffer.alloc(DATA_DIRECTORY_ENTRY_SIZE), 0, DATA_DIRECTORY_ENTRY_SIZE, offsetInFile)
  } finally {
    fs.closeSync(fd)
  }
}

/**
 * @param {string} dir
 * @returns {Generator<string>}
 */
function* walkFiles(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isSymbolicLink()) continue
    if (entry.isDirectory()) yield* walkFiles(full)
    else if (entry.isFile()) yield full
  }
}

/**
 * Repair every dangling security directory under a tree.
 *
 * @param {string} rootDir
 * @returns {{ scanned: number, repaired: string[] }} repaired paths are
 *   relative to rootDir, in walk order.
 */
export function sanitizeTree(rootDir) {
  const repaired = []
  let scanned = 0
  for (const file of walkFiles(rootDir)) {
    let entry
    try {
      entry = readSecurityDirectory(file)
    } catch {
      continue // unreadable: packaging would already have failed on it
    }
    if (!entry) continue
    scanned += 1
    if (classifySecurityDirectory(entry) !== "dangling") continue
    clearSecurityDirectory(file, entry.offsetInFile)
    repaired.push(path.relative(rootDir, file))
  }
  return { scanned, repaired }
}

function main() {
  const root = process.argv[2]
  if (!root) {
    console.error("usage: sanitize-pe-signatures.mjs <dir>")
    process.exit(2)
  }
  const { scanned, repaired } = sanitizeTree(root)
  console.log(`[sanitize-pe-signatures] ${scanned} PEs scanned, ${repaired.length} repaired`)
  for (const file of repaired) {
    console.log(`  cleared dangling certificate table: ${file}`)
  }
}

if (isMain(import.meta.url)) {
  main()
}
