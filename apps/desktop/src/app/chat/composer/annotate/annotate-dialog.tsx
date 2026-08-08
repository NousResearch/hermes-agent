import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { PencilLine } from '@/lib/icons'

import { AnnotationCanvas, type AnnotationCanvasHandle } from './annotation-canvas'
import { type AnnotationShape, shapesToLegend } from './annotation-model'

export interface AnnotateResult {
  /** Composite PNG data URL (original image + overlay rendered into one buffer). */
  dataUrl: string
  /** Auto-generated legend text, editable by the user before sending. */
  legend: string
}

interface AnnotateDialogProps {
  imageSrc: string
  imageLabel: string
  onClose: () => void
  onSave: (result: AnnotateResult) => void
  open: boolean
}

export function AnnotateDialog({ imageSrc, imageLabel, onClose, onSave, open }: AnnotateDialogProps) {
  const { t } = useI18n()
  const c = t.composer
  const canvasHandleRef = useRef<AnnotationCanvasHandle | null>(null)
  const [shapes, setShapes] = useState<AnnotationShape[]>([])

  const handleSave = () => {
    const canvas = canvasHandleRef.current?.canvas

    if (!canvas) {
      return
    }

    const dataUrl = canvas.toDataURL('image/png')
    onSave({ dataUrl, legend: shapesToLegend(shapes) })
  }

  const handleClose = () => {
    setShapes([])
    onClose()
  }

  return (
    <Dialog onOpenChange={nextOpen => void (nextOpen ? undefined : handleClose())} open={open}>
      <DialogContent className="max-w-4xl gap-5">
        <DialogHeader>
          <DialogTitle icon={PencilLine}>{c.annotateTitle}</DialogTitle>
          <DialogDescription>{c.annotateDesc(imageLabel)}</DialogDescription>
        </DialogHeader>
        <AnnotationCanvas onChange={setShapes} ref={canvasHandleRef} src={imageSrc} />
        <DialogFooter>
          <Button onClick={handleClose} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={shapes.length === 0} onClick={handleSave} type="button">
            {c.annotateSave}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
