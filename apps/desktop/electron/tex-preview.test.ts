import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  compileTexPreview,
  findTexExecutable,
  parseTexDiagnostics,
  texDirectives,
  texSubprocessEnvironment
} from './tex-preview'

test('TeX directives resolve an explicit root and allowlisted engine', () => {
  const sourcePath = path.join('/work', 'chapters', 'intro.tex')
  const directives = texDirectives('% !TeX root = ../paper.tex\n% !TeX program = lualatex\n', sourcePath)

  assert.equal(directives.rootPath, path.join('/work', 'paper.tex'))
  assert.equal(directives.program, 'lualatex')
})

test('TeX directives reject arbitrary program execution', () => {
  const directives = texDirectives('% !TeX program = bash\n', '/work/paper.tex')

  assert.equal(directives.rootPath, '/work/paper.tex')
  assert.equal(directives.program, undefined)
})

test('TeX subprocess environment excludes inherited credentials', () => {
  const env = texSubprocessEnvironment({
    HOME: '/home/test',
    OPENAI_API_KEY: 'secret',
    PATH: '/usr/bin',
    TEXMFHOME: '/home/test/texmf'
  })

  assert.equal(env.OPENAI_API_KEY, undefined)
  assert.equal(env.PATH, '/usr/bin')
  assert.equal(env.TEXMFHOME, '/home/test/texmf')
  assert.equal(env.openin_any, 'p')
})

test('TeX diagnostics retain file and line context', () => {
  assert.deepEqual(parseTexDiagnostics('/work/paper.tex:14: Undefined control sequence\n! Emergency stop.'), [
    { file: '/work/paper.tex', line: 14, message: 'Undefined control sequence' },
    { message: 'Emergency stop.' }
  ])
})

test.runIf(
  Boolean(
    findTexExecutable('latexmk') &&
    (findTexExecutable('xelatex') || findTexExecutable('lualatex') || findTexExecutable('pdflatex'))
  )
)('TeX compilation produces a real PDF outside the source directory', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-tex-test-'))
  const sourcePath = path.join(root, 'paper.tex')
  const outputRoot = path.join(root, 'output')
  fs.writeFileSync(sourcePath, '\\documentclass{article}\\begin{document}Hermes $x^2$\\end{document}\n')

  try {
    const result = await compileTexPreview({
      outputRoot,
      requestId: 'test',
      signal: new AbortController().signal,
      sourcePath
    })

    assert.equal(result.status, 'success', result.log)
    assert.ok(result.pdfPath?.startsWith(outputRoot))
    assert.ok(fs.statSync(result.pdfPath!).size > 100)
    assert.equal(fs.existsSync(path.join(root, 'paper.pdf')), false)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})
