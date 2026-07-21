import assert from 'node:assert/strict'

import { test } from 'vitest'

import { computeAvatarCrop } from './avatar-crop'

test('square image: full source mapped to target', () => {
  const p = computeAvatarCrop(512, 512, 256)
  assert.deepStrictEqual(p, {
    sx: 0,
    sy: 0,
    sWidth: 512,
    sHeight: 512,
    dWidth: 256,
    dHeight: 256
  })
})

test('landscape image: crops centered square from height', () => {
  // 4000×3000 → crop 3000×3000 from x=500
  const p = computeAvatarCrop(4000, 3000, 256)
  assert.deepStrictEqual(p, {
    sx: 500,
    sy: 0,
    sWidth: 3000,
    sHeight: 3000,
    dWidth: 256,
    dHeight: 256
  })
})

test('portrait image: crops centered square from width', () => {
  // 2000×3000 → crop 2000×2000 from y=500
  const p = computeAvatarCrop(2000, 3000, 256)
  assert.deepStrictEqual(p, {
    sx: 0,
    sy: 500,
    sWidth: 2000,
    sHeight: 2000,
    dWidth: 256,
    dHeight: 256
  })
})

test('tiny image smaller than maxSize: upscales', () => {
  // 50×50 → crops 50×50, draws at 256×256 (upscale)
  const p = computeAvatarCrop(50, 50, 256)
  assert.deepStrictEqual(p, {
    sx: 0,
    sy: 0,
    sWidth: 50,
    sHeight: 50,
    dWidth: 256,
    dHeight: 256
  })
})

test('custom maxSize respected', () => {
  const p = computeAvatarCrop(1000, 800, 128)
  assert.deepStrictEqual(p, {
    sx: 100,
    sy: 0,
    sWidth: 800,
    sHeight: 800,
    dWidth: 128,
    dHeight: 128
  })
})
