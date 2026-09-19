export function resolveLinuxUnpackedDirName(arch) {
  return arch === 'x64' ? 'linux-unpacked' : `linux-${arch}-unpacked`
}
