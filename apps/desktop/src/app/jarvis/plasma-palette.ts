import type { PlasmaPalette, PlasmaTone } from './plasma'

export const PALETTES: Record<PlasmaTone, PlasmaPalette> = {
  thinking: {
    back: '105, 110, 240', core: '205, 215, 255', front: '115, 145, 250',
    highlight: '165, 185, 255', rim: '145, 165, 245'
  },
  idle: {
    back: '124, 92, 255',
    core: '150, 225, 255',
    front: '0, 183, 255',
    highlight: '87, 210, 255',
    rim: '92, 200, 255'
  },
  listening: {
    back: '124, 92, 255',
    core: '190, 240, 255',
    front: '40, 200, 255',
    highlight: '103, 224, 255',
    rim: '120, 215, 255'
  },
  speaking: {
    back: '0, 183, 255',
    core: '200, 255, 230',
    front: '41, 230, 140',
    highlight: '105, 245, 185',
    rim: '90, 240, 190'
  },
  working: {
    back: '0, 183, 255',
    core: '205, 190, 255',
    front: '150, 120, 255',
    highlight: '180, 155, 255',
    rim: '150, 140, 255'
  },
  approval: {
    back: '124, 92, 255',
    core: '255, 236, 190',
    front: '246, 196, 83',
    highlight: '255, 222, 146',
    rim: '246, 210, 120'
  },
  success: {
    back: '0, 183, 255',
    core: '200, 255, 225',
    front: '41, 230, 140',
    highlight: '120, 245, 180',
    rim: '110, 240, 180'
  },
  error: {
    back: '124, 92, 255',
    core: '255, 200, 210',
    front: '255, 77, 109',
    highlight: '255, 145, 162',
    rim: '255, 120, 140'
  }
}
