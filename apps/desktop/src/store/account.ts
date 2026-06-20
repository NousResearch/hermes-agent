import { atom } from 'nanostores'

// Controls the EasyHermes account popup (login/register when signed out;
// balance + usage + logout + about when signed in). Opened from the titlebar
// account button.
export const $accountDialogOpen = atom(false)

export function openAccountDialog() {
  $accountDialogOpen.set(true)
}

export function closeAccountDialog() {
  $accountDialogOpen.set(false)
}

export function setAccountDialogOpen(open: boolean) {
  $accountDialogOpen.set(open)
}
