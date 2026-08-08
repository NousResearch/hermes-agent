/**
 * Pure helper for the avatar image-to-PNG pipeline.
 *
 * Given a source image dimension (width, height) and a target size,
 * returns the drawImage-compatible parameters for center-cropping the
 * largest possible square from the source and scaling it down to
 * maxSize×maxSize.
 */
export interface CropParams {
  sx: number
  sy: number
  sWidth: number
  sHeight: number
  dWidth: number
  dHeight: number
}

export function computeAvatarCrop(
  imgWidth: number,
  imgHeight: number,
  maxSize = 256
): CropParams {
  const cropSize = Math.min(imgWidth, imgHeight)
  const sx = (imgWidth - cropSize) / 2
  const sy = (imgHeight - cropSize) / 2

  return {
    sx,
    sy,
    sWidth: cropSize,
    sHeight: cropSize,
    dWidth: maxSize,
    dHeight: maxSize
  }
}
