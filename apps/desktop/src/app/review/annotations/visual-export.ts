import { attachmentId, pathLabel } from '@/lib/chat-runtime'
import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import type { ReviewAnnotation, VisualAnnotationAnchor, VisualAnnotationMark } from '@/store/annotations'
import { mainComposerScope } from '@/store/composer'

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('The annotated image could not be decoded'))
    image.src = dataUrl
  })
}

function drawMark(
  context: CanvasRenderingContext2D,
  mark: VisualAnnotationMark,
  number: number,
  width: number,
  height: number
): void {
  const point = (value: { x: number; y: number }) => ({ x: value.x * width, y: value.y * height })
  context.strokeStyle = '#ff3b5c'
  context.fillStyle = '#ff3b5c'
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = Math.max(3, Math.min(width, height) / 180)

  if (mark.tool === 'pen') {
    context.beginPath()
    mark.points.forEach((value, index) => {
      const next = point(value)

      if (index) {
        context.lineTo(next.x, next.y)
      } else {
        context.moveTo(next.x, next.y)
      }
    })
    context.stroke()

    return
  }

  if (mark.tool === 'rectangle') {
    const start = point(mark.start)
    const end = point(mark.end)
    context.fillStyle = 'rgba(255, 59, 92, 0.12)'
    context.fillRect(start.x, start.y, end.x - start.x, end.y - start.y)
    context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y)

    return
  }

  if (mark.tool === 'arrow') {
    const start = point(mark.start)
    const end = point(mark.end)
    const angle = Math.atan2(end.y - start.y, end.x - start.x)
    const head = Math.max(12, Math.min(width, height) / 35)
    context.beginPath()
    context.moveTo(start.x, start.y)
    context.lineTo(end.x, end.y)
    context.lineTo(end.x - head * Math.cos(angle - Math.PI / 6), end.y - head * Math.sin(angle - Math.PI / 6))
    context.moveTo(end.x, end.y)
    context.lineTo(end.x - head * Math.cos(angle + Math.PI / 6), end.y - head * Math.sin(angle + Math.PI / 6))
    context.stroke()

    return
  }

  const pin = point(mark.point)
  const radius = Math.max(12, Math.min(width, height) / 42)
  context.beginPath()
  context.arc(pin.x, pin.y, radius, 0, Math.PI * 2)
  context.fill()
  context.fillStyle = '#ffffff'
  context.font = `700 ${Math.round(radius * 1.25)}px system-ui`
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(String(number), pin.x, pin.y)
}

async function saveComposite(path: string, anchors: VisualAnnotationAnchor[]): Promise<string> {
  const image = await loadImage(await readDesktopFileDataUrl(path))
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Image annotation export is unavailable')
  }

  context.drawImage(image, 0, 0)
  anchors
    .flatMap(anchor => anchor.marks)
    .forEach((mark, index) => drawMark(context, mark, index + 1, canvas.width, canvas.height))

  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      value => (value ? resolve(value) : reject(new Error('Could not flatten image annotations'))),
      'image/png'
    )
  )

  const savedPath = await window.hermesDesktop.saveImageBuffer(await blob.arrayBuffer(), '.png')
  mainComposerScope.add({
    id: attachmentId('image', savedPath),
    kind: 'image',
    label: `${pathLabel(path)} — annotated`,
    path: savedPath,
    previewUrl: URL.createObjectURL(blob)
  })

  return savedPath
}

export async function attachVisualAnnotationComposites(items: readonly ReviewAnnotation[]): Promise<string[]> {
  const grouped = new Map<string, VisualAnnotationAnchor[]>()

  for (const item of items) {
    if (item.anchor.kind !== 'visual' || item.anchor.marks.length === 0) {
      continue
    }

    grouped.set(item.anchor.path, [...(grouped.get(item.anchor.path) ?? []), item.anchor])
  }

  return Promise.all([...grouped].map(([path, anchors]) => saveComposite(path, anchors)))
}
