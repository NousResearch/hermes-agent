import js from '@eslint/js'
import typescriptEslint from '@typescript-eslint/eslint-plugin'
import typescriptParser from '@typescript-eslint/parser'
import perfectionist from 'eslint-plugin-perfectionist'
import reactPlugin from 'eslint-plugin-react'
import hooksPlugin from 'eslint-plugin-react-hooks'
import unusedImports from 'eslint-plugin-unused-imports'
import globals from 'globals'

const noopRule = {
  meta: { schema: [], type: 'problem' },
  create: () => ({})
}

const LOCAL_ONLY_DESKTOP_FS_EXPORTS = new Set(['revealDesktopPath', 'renameDesktopPath', 'trashDesktopPath'])
const RAW_READ_FILE_DATA_URL_ALLOWLIST = [
  '/src/app/session/hooks/use-prompt-actions/utils.ts',
  '/src/lib/desktop-fs.ts'
]

function normalizedFilename(context) {
  return (context.filename || context.getFilename?.() || '').replace(/\\/g, '/')
}

function isAllowlistedRawReadFileDataUrl(context) {
  const filename = normalizedFilename(context)

  return RAW_READ_FILE_DATA_URL_ALLOWLIST.some(suffix => filename.endsWith(suffix))
}

function propertyName(node) {
  if (!node) return ''
  if (!node.computed && node.property?.type === 'Identifier') return node.property.name
  if (node.computed && node.property?.type === 'Literal') return String(node.property.value)

  return ''
}

function isIdentifier(node, name) {
  return node?.type === 'Identifier' && node.name === name
}

function isWindowHermesDesktopReadFileDataUrl(node) {
  return (
    propertyName(node) === 'readFileDataUrl' &&
    node.object?.type === 'MemberExpression' &&
    propertyName(node.object) === 'hermesDesktop' &&
    isIdentifier(node.object.object, 'window')
  )
}

const noLocalDesktopFileOpsRule = {
  meta: {
    messages: {
      localOnlyDesktopFs:
        "Do not import local-only desktop-fs export '{{name}}' here. Route through a remote-aware facade or guard remote mode with an honest message.",
      rawFs: 'Renderer code must not call raw fs.* APIs; route file access through the desktop filesystem facade.',
      rawReadFileDataUrl:
        'Do not call window.hermesDesktop.readFileDataUrl outside an allowlisted local-file seam; use readDesktopFileDataUrl or another remote-aware facade.',
      rawShell: 'Renderer code must not call raw shell.* APIs; route opens through the desktop bridge/facade.'
    },
    schema: [],
    type: 'problem'
  },
  create(context) {
    const fsAliases = new Set()
    const shellAliases = new Set()
    const desktopFsNamespaces = new Set()

    return {
      ImportDeclaration(node) {
        const source = String(node.source.value || '')

        if (source === '@/lib/desktop-fs') {
          for (const specifier of node.specifiers) {
            if (specifier.type === 'ImportNamespaceSpecifier') {
              desktopFsNamespaces.add(specifier.local.name)
              continue
            }

            const importedName = specifier.imported?.name || specifier.imported?.value
            if (LOCAL_ONLY_DESKTOP_FS_EXPORTS.has(importedName)) {
              context.report({ data: { name: importedName }, messageId: 'localOnlyDesktopFs', node: specifier })
            }
          }
        }

        if (source === 'fs' || source === 'node:fs') {
          for (const specifier of node.specifiers) {
            if (specifier.type === 'ImportDefaultSpecifier' || specifier.type === 'ImportNamespaceSpecifier') {
              fsAliases.add(specifier.local.name)
            }
          }
        }

        if (source === 'electron') {
          for (const specifier of node.specifiers) {
            if (specifier.type === 'ImportSpecifier' && specifier.imported?.name === 'shell') {
              shellAliases.add(specifier.local.name)
            }
          }
        }
      },
      MemberExpression(node) {
        if (node.object?.type === 'Identifier' && fsAliases.has(node.object.name)) {
          context.report({ messageId: 'rawFs', node })
          return
        }

        if (node.object?.type === 'Identifier' && shellAliases.has(node.object.name)) {
          context.report({ messageId: 'rawShell', node })
          return
        }

        if (desktopFsNamespaces.has(node.object?.name) && LOCAL_ONLY_DESKTOP_FS_EXPORTS.has(propertyName(node))) {
          context.report({ data: { name: propertyName(node) }, messageId: 'localOnlyDesktopFs', node })
          return
        }

        if (isWindowHermesDesktopReadFileDataUrl(node) && !isAllowlistedRawReadFileDataUrl(context)) {
          context.report({ messageId: 'rawReadFileDataUrl', node })
        }
      }
    }
  }
}

const customRules = {
  rules: {
    'no-local-desktop-file-ops': noLocalDesktopFileOpsRule,
    'no-process-cwd': noopRule,
    'no-process-env-top-level': noopRule,
    'no-sync-fs': noopRule,
    'no-top-level-dynamic-import': noopRule,
    'no-top-level-side-effects': noopRule
  }
}

export default [
  {
    ignores: ['**/node_modules/**', '**/dist/**', 'src/**/*.js'],
    linterOptions: { noInlineConfig: true, reportUnusedDisableDirectives: 'error' }
  },
  js.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      },
      parser: typescriptParser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
        ecmaVersion: 'latest',
        sourceType: 'module'
      }
    },
    plugins: {
      '@typescript-eslint': typescriptEslint,
      'custom-rules': customRules,
      perfectionist,
      react: reactPlugin,
      'react-hooks': hooksPlugin,
      'unused-imports': unusedImports
    },
    rules: {
      '@typescript-eslint/consistent-type-imports': ['error', { prefer: 'type-imports' }],
      '@typescript-eslint/no-unused-vars': 'off',
      'custom-rules/no-local-desktop-file-ops': 'error',
      curly: ['error', 'all'],
      'no-fallthrough': ['error', { allowEmptyCase: true }],
      'no-undef': 'off',
      'no-unused-vars': 'off',
      'padding-line-between-statements': [
        1,
        {
          blankLine: 'always',
          next: [
            'block-like',
            'block',
            'return',
            'if',
            'class',
            'continue',
            'debugger',
            'break',
            'multiline-const',
            'multiline-let'
          ],
          prev: '*'
        },
        {
          blankLine: 'always',
          next: '*',
          prev: ['case', 'default', 'multiline-const', 'multiline-let', 'multiline-block-like']
        },
        { blankLine: 'never', next: ['block', 'block-like'], prev: ['case', 'default'] },
        { blankLine: 'always', next: ['block', 'block-like'], prev: ['block', 'block-like'] },
        { blankLine: 'always', next: ['empty'], prev: 'export' },
        { blankLine: 'never', next: 'iife', prev: ['block', 'block-like', 'empty'] }
      ],
      'perfectionist/sort-exports': ['error', { order: 'asc', type: 'natural' }],
      'perfectionist/sort-imports': [
        'error',
        {
          groups: ['side-effect', 'builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
          order: 'asc',
          type: 'natural'
        }
      ],
      'perfectionist/sort-jsx-props': ['error', { order: 'asc', type: 'natural' }],
      'perfectionist/sort-named-exports': ['error', { order: 'asc', type: 'natural' }],
      'perfectionist/sort-named-imports': ['error', { order: 'asc', type: 'natural' }],
      'react-hooks/exhaustive-deps': 'warn',
      'react-hooks/rules-of-hooks': 'error',
      'unused-imports/no-unused-imports': 'error'
    },
    settings: {
      react: { version: 'detect' }
    }
  },
  {
    files: ['**/*.js', '**/*.cjs', '**/*.mjs'],
    ignores: ['**/node_modules/**', '**/dist/**'],
    languageOptions: {
      ecmaVersion: 'latest',
      globals: { ...globals.node },
      sourceType: 'module'
    }
  },
  {
    ignores: ['*.config.*']
  }
]
