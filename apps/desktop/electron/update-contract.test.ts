import { describe, expect, it } from 'vitest'
import { parseRequiredBackendContract } from './update-contract'
describe('parseRequiredBackendContract', () => {
  it('reads one valid target contract', () => expect(parseRequiredBackendContract('const REQUIRED_BACKEND_CONTRACT = 8\n')).toBe(8))
  it.each([
    '',
    'const REQUIRED_BACKEND_CONTRACT = unknown\n',
    'const REQUIRED_BACKEND_CONTRACT = 8\nconst REQUIRED_BACKEND_CONTRACT = 9\n',
    '// const REQUIRED_BACKEND_CONTRACT = 8\n',
    '/* const REQUIRED_BACKEND_CONTRACT = 8 */\n',
    "const note = 'const REQUIRED_BACKEND_CONTRACT = 8'\n"
  ])(
    'fails closed for absent or unparseable declarations', source => expect(parseRequiredBackendContract(source)).toBeNull()
  )
})