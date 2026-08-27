/**
 * The avatar editor shared by Edit Profile and New Bot: shape grid + color
 * swatches, the Generate and Upload tabs, and the petdex Pet tab.
 */

import {
  Button,
  cn,
  Codicon,
  ColorSwatches,
  GlyphSpinner,
  host,
  PROFILE_SWATCHES,
  RowButton,
  SegmentedControl,
  Textarea,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import {
  AVATAR_PICKER_SHAPES,
  avatarColor,
  BLOB_KINDS,
  blobatarSvg,
  blobShapeString,
  BotFace,
  defaultShapeFor,
  isBlobShape,
  parseBlobShape
} from './avatar'
import {
  $imagenAvailable,
  generateAvatarImage,
  type GeneratedImage,
  normalizeAvatarImage,
  pickImageFromDevice,
  probeImagen
} from './avatar-image'
import { useBots } from './i18n'
import { PetTab } from './pet'

interface AvatarPickerProps {
  /** `null` = no explicit pick, i.e. the name's deterministic hue. */
  color: null | string
  /** Feeds the Generate tab when the user leaves the description blank. */
  generateSeed?: { description?: string; name?: string; title?: string } | null
  image: null | string
  onColor: (color: null | string) => void
  onImage: (image: null | string) => void
  onShape: (shape: string) => void
  shape: string
}

/** Shape grid + color swatches, shared by Edit Profile and New Bot. */
export function AvatarPicker({ shape, color, image, onShape, onColor, onImage, generateSeed }: AvatarPickerProps) {
  const b = useBots()
  const pickerName = generateSeed?.name || 'agent'
  const imagen = useValue($imagenAvailable)
  const [tab, setTab] = useState('bot')
  const [describe, setDescribe] = useState('')
  const [genBusy, setGenBusy] = useState(false)
  if (imagen === null) {
    void probeImagen()
  }

  // Re-check a stale "unavailable" whenever the user lands on the Generate
  // tab — the gateway may have restarted with image.generate since.
  const goTab = (id: string) => {
    setTab(id)
    if (id === 'generate' && $imagenAvailable.get() === false) {
      $imagenAvailable.set(null)
      void probeImagen()
    }
  }
  const upload = async () => {
    const raw = await pickImageFromDevice()
    if (raw) {
      onImage(await normalizeAvatarImage(raw))
    }
  }
  const generate = async () => {
    if (genBusy) {
      return
    }
    setGenBusy(true)
    try {
      const custom = describe.trim()
      const img = custom
        ? await (async () => {
            const res = await host.request<GeneratedImage>('image.generate', {
              prompt: `${custom}. Avatar for an AI agent: centered, bold flat vector style, solid color background, no text.`,
              aspect_ratio: 'square'
            })
            if (!res?.success) {
              throw new Error(res?.error || 'generation failed')
            }
            return res.image_data || res.image
          })()
        : await generateAvatarImage(generateSeed?.name || 'agent', generateSeed?.title, generateSeed?.description)
      if (img) {
        onImage(await normalizeAvatarImage(img))
      }
    } catch (err) {
      host.notifyError(err, b.avatar.generationFailed)
    } finally {
      setGenBusy(false)
    }
  }
  return (
    <div className="grid justify-items-center gap-3">
      {/* Tab pills: Bot | Generate | Upload | Pet */}
      <SegmentedControl
        options={[
          { id: 'bot', label: 'Bot' },
          { id: 'generate', label: 'Generate' },
          { id: 'upload', label: b.avatar.upload },
          { id: 'pet', label: 'Pet' }
        ]}
        value={tab}
        onChange={goTab}
      />
      {image && tab !== 'generate' ? (
        <Button type="button" variant="ghost" size="sm" onClick={() => onImage(null)}>
          {b.avatar.removeImage}
        </Button>
      ) : null}
      {tab === 'bot' ? (
        isBlobShape(shape) && blobatarSvg ? (
          (() => {
            const { seedPart, kind } = parseBlobShape(shape, pickerName)
            const locked = Boolean(seedPart)
            return (
              <div className="grid justify-items-center gap-3">
                {/* Silhouette pins: Auto (name decides) + the six blob kinds. */}
                <div className="grid grid-cols-4 justify-items-center gap-1.5">
                  {['', ...BLOB_KINDS].map(k => (
                    <RowButton
                      key={k || 'auto'}
                      title={k || 'Auto — the name decides'}
                      className={cn(
                        'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                        k === kind && !image && 'ring-1 ring-(--ui-accent)'
                      )}
                      style={{
                        width: 44,
                        height: 44
                      }}
                      onClick={() => {
                        onImage(null)
                        onShape(blobShapeString(seedPart, k))
                      }}
                    >
                      {k ? (
                        <BotFace shape={blobShapeString(seedPart, k)} color={avatarColor(color, pickerName)} size={32} name={pickerName} />
                      ) : (
                        <span className="text-[0.6rem] text-(--ui-text-tertiary)">Auto</span>
                      )}
                    </RowButton>
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      onImage(null)
                      onShape(blobShapeString(Math.random().toString(36).slice(2, 10), kind))
                    }}
                  >
                    <Codicon name="refresh" className="mr-1 text-[0.8rem]" />
                    {b.avatar.randomize}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    title={
                      locked ? b.avatar.unlockFollowsName : 'Keep this exact face even if the name changes'
                    }
                    onClick={() => onShape(blobShapeString(locked ? '' : pickerName, kind))}
                  >
                    <Codicon name={locked ? 'unlock' : 'lock'} className="mr-1 text-[0.8rem]" />
                    {locked ? 'Unlock' : 'Lock face'}
                  </Button>
                </div>
                <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">
                  {locked ? 'Face locked — renaming won\u2019t change it.' : 'Face follows the name.'}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="text-(--ui-text-tertiary)"
                  onClick={() => onShape(defaultShapeFor(pickerName))}
                >
                  {b.avatar.classicShapes}
                </Button>
              </div>
            )
          })()
        ) : (
          <div className="grid justify-items-center gap-3">
            <div className="grid grid-cols-4 justify-items-center gap-1.5">
              {(blobatarSvg ? ['blobatar', ...AVATAR_PICKER_SHAPES] : AVATAR_PICKER_SHAPES).map(s => (
                <RowButton
                  key={s}
                  title={s === 'blobatar' ? b.avatar.blobFromName : undefined}
                  className={cn(
                    'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                    s === shape && !image && 'ring-1 ring-(--ui-accent)'
                  )}
                  style={{
                    width: 44,
                    height: 44
                  }}
                  onClick={() => {
                    onImage(null)
                    onShape(s)
                  }}
                >
                  <BotFace shape={s} color={avatarColor(color, pickerName)} size={32} name={pickerName} />
                </RowButton>
              ))}
            </div>
            <ColorSwatches
              clearLabel={b.avatar.matchTheName}
              onChange={onColor}
              swatches={PROFILE_SWATCHES}
              value={color}
            />
          </div>
        )
      ) : null}
      {tab === 'generate' ? (
        imagen ? (
          <div className="grid w-full gap-2">
            <Textarea
              className="min-h-16 text-xs"
              placeholder={b.avatar.describePlaceholder}
              value={describe}
              onChange={event => setDescribe(event.target.value)}
            />
            <Button
              type="button"
              variant="secondary"
              className="w-full justify-center"
              disabled={genBusy}
              onClick={generate}
            >
              {genBusy ? (
                <GlyphSpinner spinner="breathe" className="mr-1 text-[0.8rem]" />
              ) : (
                <Codicon name="sparkle" className="mr-1 text-[0.8rem]" />
              )}
              {genBusy ? 'Generating…' : 'Generate'}
            </Button>
            {describe.trim() ? null : (
              <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">
                {b.bot.descriptionHint}
              </div>
            )}
          </div>
        ) : (
          <div className="px-2 py-3 text-center text-xs leading-5 text-(--ui-text-tertiary)">
            {imagen === false
              ? 'No image model available. If you just enabled one (or updated Hermes), restart the gateway: Ctrl+K → "Restart gateway".'
              : 'Checking image backend…'}
          </div>
        )
      ) : null}
      {tab === 'upload' ? (
        <Button type="button" variant="secondary" className="w-full justify-center" onClick={upload}>
          <Codicon name="device-camera" className="mr-1 text-[0.8rem]" />
          Choose an image…
        </Button>
      ) : null}
      {tab === 'pet' ? <PetTab image={image} onImage={onImage} /> : null}
    </div>
  )
}
