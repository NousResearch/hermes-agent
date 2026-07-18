import { DESKTOP_SOURCE_EXTENSIONS } from '../../vite.config'

describe('desktop source resolution', () => {
  it.each([
    ['.mts', '.mjs'],
    ['.ts', '.js'],
    ['.tsx', '.jsx']
  ])('prefers %s sources over stale %s siblings', (source, generated) => {
    expect(DESKTOP_SOURCE_EXTENSIONS.indexOf(source)).toBeLessThan(
      DESKTOP_SOURCE_EXTENSIONS.indexOf(generated)
    )
  })
})
