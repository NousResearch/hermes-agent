import { useMediaQuery } from './use-media-query'

export const useIsMobile = () => useMediaQuery(`(max-width: ${768 / 16 - 1 / 16}rem)`)
