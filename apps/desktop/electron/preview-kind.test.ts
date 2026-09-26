import { describe, expect, it } from 'vitest'

import { previewKindForFile } from './preview-kind'

/**
 * Regression for #audio-preview: selecting a `.wav` in the Desktop file
 * browser showed "This looks like a binary file".
 *
 * Root cause: the main-process classifier had three media branches
 * (html/image/pdf) and fell through to `metadata.binary ? 'binary' : 'text'`.
 * An audio file is sniffed binary, so it was refused — even though Desktop
 * owns a Range-capable `hermes-media://` stream protocol and the transcript
 * already renders `<audio controls>` for exactly these paths.
 */

function classify(path: string, options: { binary?: boolean; mimeType?: string } = {}) {
  return previewKindForFile({
    binary: options.binary ?? true,
    extension: extensionOf(path),
    mimeType: options.mimeType ?? mimeForExtension(path)
  })
}

function extensionOf(path: string): string {
  const index = path.lastIndexOf('.')

  return index >= 0 ? path.slice(index).toLowerCase() : ''
}

function mimeForExtension(path: string): string {
  const extension = extensionOf(path)

  if (extension === '.html') {
    return 'text/html'
  }

  if (extension === '.png') {
    return 'image/png'
  }

  if (extension === '.pdf') {
    return 'application/pdf'
  }

  if (['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'].includes(extension)) {
    return `audio/${extension.slice(1)}`
  }

  return 'application/octet-stream'
}

describe('previewKindForFile', () => {
  // The user-visible bug: the sniff says binary, and the classifier must
  // out-vote it for a format the rail can actually play.
  it.each(['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'])(
    'classifies a binary-sniffed %s as audio',
    extension => {
      expect(classify(`/home/me/take${extension}`, { binary: true })).toBe('audio')
    }
  )

  it('still refuses a genuinely opaque binary', () => {
    expect(classify('/home/me/archive.zip', { binary: true, mimeType: 'application/zip' })).toBe('binary')
  })

  it('keeps the html, image and pdf branches intact', () => {
    expect(classify('/srv/index.html', { binary: false })).toBe('html')
    expect(classify('/tmp/shot.png', { binary: false })).toBe('image')
    expect(classify('/tmp/paper.pdf', { binary: true, mimeType: 'application/pdf' })).toBe('pdf')
  })

  it('still classifies plain text as text', () => {
    expect(classify('/home/me/notes.txt', { binary: false, mimeType: 'text/plain' })).toBe('text')
  })

  // A video file is playable media but has no rail affordance yet: it must not
  // be handed to the audio player, and it must keep today's behaviour.
  it('does not classify a video file as audio', () => {
    expect(classify('/home/me/clip.mp4', { binary: true, mimeType: 'video/mp4' })).not.toBe('audio')
  })

  // Mime is the stronger signal when the name has no usable extension (an
  // extension-less recording, or a path behind a query string).
  it('trusts an audio mime type when the extension says nothing', () => {
    expect(classify('/home/me/recording', { binary: true, mimeType: 'audio/wav' })).toBe('audio')
  })
})
