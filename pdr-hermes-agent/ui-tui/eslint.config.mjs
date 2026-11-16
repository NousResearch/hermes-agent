import shared from '../eslint.config.shared.mjs'

export default [
  ...shared,
  {
    files: ['packages/hermes-ink/**/*.{ts,tsx}'],
    rules: {
      '@typescript-eslint/consistent-type-imports': 'off',
      'no-constant-condition': 'off',
      'no-empty': 'off',
      'no-redeclare': 'off',
      'react-hooks/exhaustive-deps': 'off'
    }
  }
]
