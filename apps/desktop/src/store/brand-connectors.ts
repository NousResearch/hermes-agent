import { atom } from 'nanostores'

import type { DesktopBrandConnectorList } from '@/global'

export const emptyBrandConnectors = (appId = ''): DesktopBrandConnectorList => ({
  appId,
  connectors: [],
  catalog: [],
  error: '',
  webSaved: false
})

export const $brandConnectors = atom<DesktopBrandConnectorList>(emptyBrandConnectors())
