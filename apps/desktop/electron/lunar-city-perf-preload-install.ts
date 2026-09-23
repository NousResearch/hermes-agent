import { contextBridge, ipcRenderer } from 'electron'

import { createLunarCityPerfPreload } from './lunar-city-perf-preload'

export function installLunarCityPerfPreload() {
  const lunarCityPerfChannel = new MessageChannel()
  const lunarCityPerf = createLunarCityPerfPreload(ipcRenderer, lunarCityPerfChannel.port1)

  if (lunarCityPerf) {
    contextBridge.exposeInMainWorld('__LUNAR_CITY_PERF_AUTHORIZED__', true)
    window.addEventListener(
      'DOMContentLoaded',
      () => {
        window.postMessage({ type: 'hermes:lunar-city-perf-runtime-port-v1' }, '*', [lunarCityPerfChannel.port2])
      },
      { once: true }
    )
    void lunarCityPerf.ready.then(() => {
      contextBridge.exposeInMainWorld('__LUNAR_CITY_PERF__', lunarCityPerf.surface)
    })
  } else {
    lunarCityPerfChannel.port1.close()
    lunarCityPerfChannel.port2.close()
  }
}
