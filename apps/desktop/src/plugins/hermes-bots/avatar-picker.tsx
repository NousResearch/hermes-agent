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
      <SegmentedControl
        onChange={goTab}
        options={[
          { id: 'bot', label: b.avatar.tabBot },
          { id: 'generate', label: b.avatar.tabGenerate },
          { id: 'upload', label: b.avatar.upload },
          { id: 'pet', label: b.avatar.tabPet }
        ]}
        value={tab}
      />
      {image && tab !== 'generate' ? (
        <Button onClick={() => onImage(null)} size="sm" type="button" variant="ghost">
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
                      className={cn(
                        'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                        k === kind && !image && 'ring-1 ring-(--ui-accent)'
                      )}
                      key={k || 'auto'}
                      onClick={() => {
                        onImage(null)
                        onShape(blobShapeString(seedPart, k))
                      }}
                      style={{
                        width: 44,
                        height: 44
                      }}
                      title={k || 'Auto — the name decides'}
                    >
                      {k ? (
                        <BotFace
                          color={avatarColor(color, pickerName)}
                          name={pickerName}
                          shape={blobShapeString(seedPart, k)}
                          size={32}
                        />
                      ) : (
                        <span className="text-[0.6rem] text-(--ui-text-tertiary)">Auto</span>
                      )}
                    </RowButton>
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    onClick={() => {
                      onImage(null)
                      onShape(blobShapeString(Math.random().toString(36).slice(2, 10), kind))
                    }}
                    size="sm"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon className="mr-1 text-[0.8rem]" name="refresh" />
                    {b.avatar.randomize}
                  </Button>
                  <Button
                    onClick={() => onShape(blobShapeString(locked ? '' : pickerName, kind))}
                    size="sm"
                    title={locked ? b.avatar.unlockFollowsName : 'Keep this exact face even if the name changes'}
                    type="button"
                    variant="ghost"
                  >
                    <Codicon className="mr-1 text-[0.8rem]" name={locked ? 'unlock' : 'lock'} />
                    {locked ? 'Unlock' : 'Lock face'}
                  </Button>
                </div>
                <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">
                  {locked ? 'Face locked — renaming won\u2019t change it.' : 'Face follows the name.'}
                </div>
                <Button
                  className="text-(--ui-text-tertiary)"
                  onClick={() => onShape(defaultShapeFor(pickerName))}
                  size="sm"
                  type="button"
                  variant="ghost"
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
                  className={cn(
                    'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                    s === shape && !image && 'ring-1 ring-(--ui-accent)'
                  )}
                  key={s}
                  onClick={() => {
                    onImage(null)
                    onShape(s)
                  }}
                  style={{
                    width: 44,
                    height: 44
                  }}
                  title={s === 'blobatar' ? b.avatar.blobFromName : undefined}
                >
                  <BotFace color={avatarColor(color, pickerName)} name={pickerName} shape={s} size={32} />
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
              onChange={event => setDescribe(event.target.value)}
              placeholder={b.avatar.describePlaceholder}
              value={describe}
            />
            <Button
              className="w-full justify-center"
              disabled={genBusy}
              onClick={generate}
              type="button"
              variant="secondary"
            >
              {genBusy ? (
                <GlyphSpinner className="mr-1 text-[0.8rem]" spinner="breathe" />
              ) : (
                <Codicon className="mr-1 text-[0.8rem]" name="sparkle" />
              )}
              {genBusy ? 'Generating…' : 'Generate'}
            </Button>
            {describe.trim() ? null : (
              <div className="text-center text-[0.65rem] text-(--ui-text-quaternary)">{b.bot.descriptionHint}</div>
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
        <Button className="w-full justify-center" onClick={upload} type="button" variant="secondary">
          <Codicon className="mr-1 text-[0.8rem]" name="device-camera" />
          Choose an image…
        </Button>
      ) : null}
      {tab === 'pet' ? <PetTab image={image} onImage={onImage} /> : null}
    </div>
  )
}
