// Standalone remote client packaging. It reuses the normal Desktop file list
// and lifecycle hooks, but has a distinct identity/output directory and never
// ships the install stamp that local bootstrap consumes.
const desktopPackage = require('./package.json')

const baseBuild = { ...desktopPackage.build }
const extraResources = baseBuild.extraResources || []

delete baseBuild.extraResources
delete baseBuild.protocols

module.exports = {
  ...baseBuild,
  appId: 'com.nousresearch.hermes.remote',
  productName: 'Hermes Remote',
  executableName: 'hermes-remote',
  artifactName: 'Hermes-Remote-${version}-${os}-${arch}.${ext}',
  directories: {
    ...baseBuild.directories,
    output: 'release/remote'
  },
  extraMetadata: {
    name: 'hermes-remote',
    description: 'Standalone desktop client for remote Hermes Agent instances.',
    desktopName: 'Hermes Remote',
    productName: 'Hermes Remote'
  },
  extraResources: extraResources.filter(resource => resource.to !== 'install-stamp.json'),
  mac: {
    ...baseBuild.mac,
    extendInfo: {
      ...baseBuild.mac?.extendInfo,
      CFBundleDisplayName: 'Hermes Remote',
      CFBundleExecutable: 'hermes-remote',
      CFBundleName: 'Hermes Remote'
    }
  },
  dmg: {
    ...baseBuild.dmg,
    title: 'Install Hermes Remote'
  },
  nsis: {
    ...baseBuild.nsis,
    shortcutName: 'Hermes Remote',
    uninstallDisplayName: 'Hermes Remote'
  },
  linux: {
    ...baseBuild.linux,
    syncDesktopName: true,
    synopsis: 'Standalone desktop client for remote Hermes Agent instances.',
    target: ['AppImage', 'flatpak']
  },
  flatpak: {
    baseVersion: '24.08',
    runtime: 'org.freedesktop.Platform',
    sdk: 'org.freedesktop.Sdk',
    runtimeVersion: '24.08',
    branch: 'stable',
    useWaylandFlags: true,
    finishArgs: [
      '--share=network',
      '--share=ipc',
      '--socket=wayland',
      '--socket=fallback-x11',
      '--socket=pulseaudio',
      '--device=dri',
      '--talk-name=org.freedesktop.FileManager1',
      '--talk-name=org.freedesktop.Notifications',
      '--talk-name=org.freedesktop.secrets'
    ]
  }
}
