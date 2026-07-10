import shared from '../../eslint.config.shared.mjs'

export default [
  ...shared,
  {
    files: ['**/*.test.tsx'],
    rules: {
      'no-restricted-globals': ['warn', 'document']
    }
  }
]
