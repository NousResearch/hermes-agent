import { type Codec, persistentAtom } from '@/lib/persisted'

export const PICKER_STYLES = ['classic', 'split', 'unified', 'separated'] as const
export type PickerStyle = (typeof PICKER_STYLES)[number]
export interface PickerStyleBehavior {
  separateReasoningPill: boolean
  modelSubmenu: 'preset-only' | 'select-and-apply' | 'hidden'
  effortInModelLabel: boolean
}

const BEHAVIORS: Record<PickerStyle, PickerStyleBehavior> = {
  classic: { separateReasoningPill: false, modelSubmenu: 'preset-only', effortInModelLabel: true },
  split: { separateReasoningPill: true, modelSubmenu: 'preset-only', effortInModelLabel: false },
  unified: { separateReasoningPill: false, modelSubmenu: 'select-and-apply', effortInModelLabel: true },
  separated: { separateReasoningPill: true, modelSubmenu: 'hidden', effortInModelLabel: false }
}

export const DEFAULT_PICKER_STYLE: PickerStyle = 'split'

const codec: Codec<PickerStyle> = {
  decode: raw => PICKER_STYLES.find(style => style === raw) ?? DEFAULT_PICKER_STYLE,
  encode: value => value
}

// Global renderer presentation preference, shared by primary, tiles and catalogs.
export const $pickerStyle = persistentAtom('hermes.desktop.pickerStyle', DEFAULT_PICKER_STYLE, codec)
export const pickerBehavior = (style: PickerStyle): PickerStyleBehavior => BEHAVIORS[style]

export function setPickerStyle(style: PickerStyle): void {
  $pickerStyle.set(style)
}
