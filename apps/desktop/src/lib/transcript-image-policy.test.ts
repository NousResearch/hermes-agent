import { expect, it } from 'vitest'

import { transcriptBase64, transcriptImage } from './transcript-media'

it.each([
  { format: 'odt' as const, width: 8192, accepted: true },
  { format: 'odt' as const, width: 8193, accepted: false },
  { format: undefined, width: 16384, accepted: true },
  { format: undefined, width: 16385, accepted: false }
])('preserves JPEG dimension boundary $width for $format', ({ format, width, accepted }) => {
  const bytes = new Uint8Array(40)
  bytes.set([255, 216, 255, 192, 0, 17, 8, 0, 1, width >> 8, width & 255], 0)
  bytes.set([255, 218, 0, 8], 21) // Scan header following the frame.
  bytes.set([255, 217], 38)
  const image = transcriptImage(`data:image/jpeg;base64,${transcriptBase64(bytes)}`, format)

  if (accepted) {
    expect(image).toMatchObject({ width, height: 1, mime: 'image/jpeg', extension: 'jpg', bytes })
  } else {
    expect(image).toBeUndefined()
  }
})

it('keeps ODT JPEG scan requirements when sharing the image reader', () => {
  const bytes = new Uint8Array(40)
  bytes.set([255, 216, 255, 192, 0, 17, 8, 0, 1, 0, 1], 0)
  bytes.set([255, 217], 38)
  const data = `data:image/jpeg;base64,${transcriptBase64(bytes)}`

  expect(transcriptImage(data)).toMatchObject({ width: 1, height: 1 })
  expect(transcriptImage(data, 'odt')).toBeUndefined()
})

it('keeps the narrower ODT dimension limit without restricting other formats', () => {
  const bytes = new Uint8Array(35)
  bytes.set([71, 73, 70, 56, 57, 97, 40, 35, 1, 0], 0) // 9000 by 1
  bytes[13] = 44
  bytes[34] = 59
  const data = `data:image/gif;base64,${transcriptBase64(bytes)}`

  expect(transcriptImage(data)).toMatchObject({ width: 9000, height: 1 })
  expect(transcriptImage(data, 'odt')).toBeUndefined()
})
