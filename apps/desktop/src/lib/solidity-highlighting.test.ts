import type { BundledLanguage } from 'shiki'
import { codeToTokens } from 'shiki'
import { describe, expect, it } from 'vitest'

import { shikiLanguageForFilename } from './markdown-code'

const source = 'pragma solidity ^0.8.24;\ncontract Demo { uint256 public count = 1; }'

describe('Solidity file highlighting', () => {
  it.each(['contracts/core/DAO.sol', 'contracts/DAO.SOL', 'C:\\contracts\\DAO.sol'])(
    'selects a working Solidity grammar for %s',
    async path => {
      const language = shikiLanguageForFilename(path)

      expect(language).toBe('solidity')

      const { tokens } = await codeToTokens(source, { lang: language as BundledLanguage, theme: 'github-dark' })
      const flat = tokens.flat()
      const keyword = flat.find(token => token.content === 'contract')
      const type = flat.find(token => token.content === 'uint256')

      expect(tokens.map(line => line.map(token => token.content).join('')).join('\n')).toBe(source)
      expect(keyword?.color).toBeTruthy()
      expect(type?.color).toBeTruthy()
      expect(keyword?.color).not.toBe(type?.color)
    }
  )
})
