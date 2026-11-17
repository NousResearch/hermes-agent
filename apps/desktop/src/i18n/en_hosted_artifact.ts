import type { HostedArtifactCopy } from './types_hosted_artifact'

export const enHostedArtifact: HostedArtifactCopy = {
  untitled: 'Artifact',
  version: version => `Version ${version}`,
  newerVersion: version => `Version ${version} available`,
  openViewer: 'Open in viewer',
  linkOnly: 'Open the viewer to see this artifact.',
  frameTitle: title => `Artifact: ${title}`,
  messagePrompt: 'The artifact wants to send this message:',
  messageSend: 'Send',
  messageDismiss: 'Dismiss'
}
