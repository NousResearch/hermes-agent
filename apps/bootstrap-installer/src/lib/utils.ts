import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/*
 * cn — Tailwind-aware class merger. Same util the desktop and dashboard
 * use. clsx handles conditional classes; twMerge resolves utility
 * conflicts so `cn('px-2', condition && 'px-4')` ends up with px-4 only,
 * not both.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
