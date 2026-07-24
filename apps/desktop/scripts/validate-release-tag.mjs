#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'

import { isMain } from './utils.mjs'

export function expectedDesktopTag(version) {
  return `desktop-v${version}`
}

export function validateDesktopReleaseTag(tag, version) {
  const expected = expectedDesktopTag(version)
  if (tag !== expected) {
    throw new Error(`Desktop release tag must be ${expected}; received ${tag || '<empty>'}`)
  }
  return expected
}

if (isMain(import.meta.url)) {
  const packageJson = JSON.parse(fs.readFileSync(path.resolve(import.meta.dirname, '..', 'package.json'), 'utf8'))

  try {
    const tag = process.argv[2] || process.env.GITHUB_REF_NAME || ''
    console.log(`[release] validated ${validateDesktopReleaseTag(tag, packageJson.version)}`)
  } catch (error) {
    console.error(`[release] ${error.message}`)
    process.exit(1)
  }
}
