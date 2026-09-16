import { createContext } from 'react'

/** The owning session's working directory — MediaAttachment uses it to resolve
 *  relative MEDIA paths (provided by Thread, so split panes stay independent). */
export const MediaCwdContext = createContext<null | string>(null)
