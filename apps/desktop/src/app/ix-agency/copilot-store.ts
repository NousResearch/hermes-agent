import { atom } from 'nanostores'

import type { IxSkillItem } from './types'

// Hand-off from the Org skills tab's "Run natively" button to the Copilot
// tab: the picked skill pre-selects its chip (playbook prompt) and seeds the
// composer with its first starter prompt.
export const $ixPendingSkill = atom<IxSkillItem | null>(null)
