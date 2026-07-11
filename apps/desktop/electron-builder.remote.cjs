const desktopPackage = require('./package.json')

const { extraResources = [], protocols: _protocols, ...baseBuild } = desktopPackage.build

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
    description: 'Standalone desktop client for remote Hermes Agent instances.',
    desktopName: 'Hermes Remote',
    productName: 'Hermes Remote'
  },
  extraResources: extraResources.filter(resource => resource.to !== 'install-stamp.json'),
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
