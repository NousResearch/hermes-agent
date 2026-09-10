import { describe, expect, it } from 'vitest'

import { cleanPath, comparisonPath, isUnderPath } from './path-compare'

describe('cleanPath', () => {
  it('unifies separators and drops trailing slashes', () => {
    expect(cleanPath('C:\\Repos\\App\\')).toBe('C:/Repos/App')
    expect(cleanPath('  /home/user/repo//  ')).toBe('/home/user/repo')
  })

  it('keeps root rather than collapsing to empty', () => {
    expect(cleanPath('/')).toBe('/')
  })

  it('collapses repeated separators', () => {
    expect(cleanPath('C:/Repos//App')).toBe('C:/Repos/App')
    expect(cleanPath('C:\\Repos\\\\App')).toBe('C:/Repos/App')
    expect(cleanPath('/home/u//repo')).toBe('/home/u/repo')
  })

  it('keeps the leading double separator of a UNC path', () => {
    expect(cleanPath('//server/share/proj')).toBe('//server/share/proj')
    expect(cleanPath('\\\\server\\share\\proj')).toBe('//server/share/proj')
    expect(cleanPath('//server//share')).toBe('//server/share')
  })
})

describe('comparisonPath', () => {
  it('folds case for Windows drive and UNC paths only', () => {
    expect(comparisonPath('C:/Repos/App')).toBe('c:/repos/app')
    expect(comparisonPath('//server/Share')).toBe('//server/share')
    expect(comparisonPath('/home/User/Repo')).toBe('/home/User/Repo')
  })
})

describe('isUnderPath', () => {
  it('matches a nested path across separator and case differences', () => {
    expect(isUnderPath('C:\\Repos\\App', 'c:/repos/app/src')).toBe(true)
    expect(isUnderPath('C:/Repos/App/', 'C:\\Repos\\App')).toBe(true)
  })

  it('stays case-sensitive on POSIX', () => {
    expect(isUnderPath('/home/user/repo', '/home/user/repo/src')).toBe(true)
    expect(isUnderPath('/home/user/repo', '/home/user/Repo/src')).toBe(false)
  })

  it('does not treat a sibling with a shared prefix as nested', () => {
    expect(isUnderPath('/repos/app', '/repos/app-retry')).toBe(false)
  })

  it('matches when either side carries a doubled separator', () => {
    // A root that already ends in a separator, joined with a child.
    expect(isUnderPath('C:/Repos//App', 'C:/Repos/App/src')).toBe(true)
    expect(isUnderPath('C:/Repos/App', 'C:/Repos//App/src')).toBe(true)
    expect(isUnderPath('C:\\Repos\\\\App', 'c:/repos/app/src')).toBe(true)
    expect(isUnderPath('/home/u//repo', '/home/u/repo/src')).toBe(true)
  })

  it('still matches under a UNC root', () => {
    expect(isUnderPath('//server/share/proj', '//SERVER/share/proj/src')).toBe(true)
    expect(isUnderPath('\\\\server\\share\\proj', '//server/share/proj')).toBe(true)
  })
})
