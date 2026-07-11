import { useCallback, useEffect, useMemo, useState } from 'react'

import type {
  MoneyPrinterResponse,
  VideoAspect,
  VideoLibraryAnalysisResult,
  VideoLibraryAsset,
  VideoLibraryClip,
  VideoLibraryClipQuery,
  VideoLibraryDescriptor,
  VideoLibraryMigrationResult,
  VideoLibraryStatus,
  VideoLibraryTimelineResult
} from './moneyprinter-client'
import {
  clearLibraryMatches,
  confirmSegmentClip,
  emptyMatchState,
  type NamedLibraryMatchState,
  planAutomaticClipPool,
  segmentVideoScript,
  setSegmentCandidates,
  setSegmentError
} from './named-library-matching'

const AUTO_MATCH_CANDIDATE_LIMIT = 12

export interface NamedVideoLibraryClient {
  addSourceRoot(
    libraryId: string,
    path: string
  ): Promise<MoneyPrinterResponse<{ library_id: string; source_roots: string[] }>>
  analyzeAsset(libraryId: string, assetId: string): Promise<MoneyPrinterResponse<VideoLibraryAnalysisResult>>
  createTimeline(
    libraryId: string,
    clipIds: string[],
    aspect: VideoAspect,
    script?: Array<Record<string, unknown>>
  ): Promise<MoneyPrinterResponse<VideoLibraryTimelineResult>>
  getLibraryStatus(libraryId: string): Promise<MoneyPrinterResponse<VideoLibraryStatus>>
  importAsset(libraryId: string, sourcePath: string): Promise<MoneyPrinterResponse<{ asset: VideoLibraryAsset }>>
  listAssets(libraryId?: string): Promise<MoneyPrinterResponse<{ assets: VideoLibraryAsset[]; total: number }>>
  listClips(
    libraryId?: string,
    options?: VideoLibraryClipQuery
  ): Promise<MoneyPrinterResponse<{ clips: VideoLibraryClip[]; total: number }>>
  listLibraries(): Promise<MoneyPrinterResponse<{ libraries: VideoLibraryDescriptor[] }>>
  migrateLegacyLibrary(libraryId: string): Promise<MoneyPrinterResponse<VideoLibraryMigrationResult>>
  scanLibrary(libraryId: string, dryRun?: boolean): Promise<MoneyPrinterResponse<Record<string, unknown>>>
}

interface UseNamedVideoLibraryOptions {
  client: NamedVideoLibraryClient
  script: string
  terms?: string
}

function requireData<T>(response: MoneyPrinterResponse<T>, fallback: string): T {
  if (!response.ok || !response.data) {
    throw new Error(response.error?.message || fallback)
  }

  return response.data
}

export function useNamedVideoLibrary({ client, script, terms = '' }: UseNamedVideoLibraryOptions) {
  const [libraries, setLibraries] = useState<VideoLibraryDescriptor[]>([])
  const [selectedLibraryId, setSelectedLibraryId] = useState('')
  const [status, setStatus] = useState<VideoLibraryStatus | null>(null)
  const [assets, setAssets] = useState<VideoLibraryAsset[]>([])
  const [clips, setClips] = useState<VideoLibraryClip[]>([])
  const [matches, setMatches] = useState<NamedLibraryMatchState>(() => emptyMatchState())
  const [error, setError] = useState('')
  const [loadingLibraries, setLoadingLibraries] = useState(true)
  const [loadingLibrary, setLoadingLibrary] = useState(false)
  const [matchingAll, setMatchingAll] = useState(false)
  const [matchingSegmentId, setMatchingSegmentId] = useState('')
  const [scanBusy, setScanBusy] = useState(false)
  const [scanPreview, setScanPreview] = useState<Record<string, unknown> | null>(null)
  const [migrationResult, setMigrationResult] = useState<VideoLibraryMigrationResult | null>(null)
  const [managementBusy, setManagementBusy] = useState(false)
  const [timelineBusy, setTimelineBusy] = useState(false)
  const [timeline, setTimeline] = useState<VideoLibraryTimelineResult | null>(null)
  const segments = useMemo(() => segmentVideoScript(script), [script])

  useEffect(() => {
    let active = true
    setLoadingLibraries(true)
    void client
      .listLibraries()
      .then(response => requireData(response, '无法加载视频资产库'))
      .then(data => {
        if (active) {setLibraries(data.libraries)}
      })
      .catch(reason => {
        if (active) {setError(reason instanceof Error ? reason.message : String(reason))}
      })
      .finally(() => {
        if (active) {setLoadingLibraries(false)}
      })

    return () => {
      active = false
    }
  }, [client])

  const refreshSelectedLibrary = useCallback(async () => {
    if (!selectedLibraryId) {return}
    setLoadingLibrary(true)
    setError('')

    try {
      const [statusResponse, assetsResponse, clipsResponse] = await Promise.all([
        client.getLibraryStatus(selectedLibraryId),
        client.listAssets(selectedLibraryId),
        client.listClips(selectedLibraryId)
      ])

      setStatus(requireData(statusResponse, '无法读取资产库状态'))
      setAssets(requireData(assetsResponse, '无法读取资产').assets)
      setClips(requireData(clipsResponse, '无法读取镜头').clips)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setLoadingLibrary(false)
    }
  }, [client, selectedLibraryId])

  useEffect(() => {
    if (selectedLibraryId) {void refreshSelectedLibrary()}
  }, [refreshSelectedLibrary, selectedLibraryId])

  const selectLibrary = useCallback((libraryId: string) => {
    setSelectedLibraryId(libraryId)
    setStatus(null)
    setAssets([])
    setClips([])
    setMatches(current => clearLibraryMatches(current))
    setTimeline(null)
    setScanPreview(null)
    setMigrationResult(null)
    setError('')
  }, [])

  const matchSegment = useCallback(
    async (segmentId: string) => {
      if (!selectedLibraryId) {throw new Error('请先选择资产库')}
      const segment = segments.find(item => item.id === segmentId)

      if (!segment) {throw new Error('文案片段不存在')}
      setMatchingSegmentId(segmentId)

      try {
        const query = [segment.text, terms.trim()].filter(Boolean).join(' ')
        const response = await client.listClips(selectedLibraryId, { limit: 5, query })
        const candidates = requireData(response, '镜头匹配失败').clips

        if (candidates.length === 0) {
          setMatches(current => setSegmentError(current, segmentId, '未找到合适镜头'))

          return []
        }

        setMatches(current => setSegmentCandidates(current, segmentId, candidates))

        return candidates
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : String(reason)
        setMatches(current => setSegmentError(current, segmentId, message))

        return []
      } finally {
        setMatchingSegmentId('')
      }
    },
    [client, segments, selectedLibraryId, terms]
  )

  const matchAll = useCallback(async () => {
    if (!selectedLibraryId) {throw new Error('请先选择资产库')}
    setMatchingAll(true)

    try {
      await Promise.allSettled(segments.map(segment => matchSegment(segment.id)))
    } finally {
      setMatchingAll(false)
    }
  }, [matchSegment, segments, selectedLibraryId])

  const confirmClip = useCallback((segmentId: string, clipId: string) => {
    setMatches(current => confirmSegmentClip(current, segmentId, clipId))
  }, [])

  const scanSelectedLibrary = useCallback(async () => {
    if (!selectedLibraryId) {throw new Error('请先选择资产库')}
    setScanBusy(true)

    try {
      const response = await client.scanLibrary(selectedLibraryId, false)
      requireData(response, '资产库扫描失败')
      await refreshSelectedLibrary()
    } finally {
      setScanBusy(false)
    }
  }, [client, refreshSelectedLibrary, selectedLibraryId])

  const importFiles = useCallback(
    async (sourcePaths: string[]) => {
      if (!selectedLibraryId) {throw new Error('请先选择资产库')}
      setManagementBusy(true)
      setError('')

      try {
        for (const sourcePath of sourcePaths) {
          const imported = requireData(
            await client.importAsset(selectedLibraryId, sourcePath),
            `素材导入失败：${sourcePath}`
          )

          requireData(await client.analyzeAsset(selectedLibraryId, imported.asset.id), `素材分析失败：${sourcePath}`)
        }

        await refreshSelectedLibrary()
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : String(reason)
        setError(message)
        throw reason
      } finally {
        setManagementBusy(false)
      }
    },
    [client, refreshSelectedLibrary, selectedLibraryId]
  )

  const addSourceRoot = useCallback(
    async (path: string) => {
      if (!selectedLibraryId) {throw new Error('请先选择资产库')}
      setManagementBusy(true)
      setError('')

      try {
        const rooted = requireData(await client.addSourceRoot(selectedLibraryId, path), '素材目录添加失败')
        setLibraries(current =>
          current.map(library =>
            library.id === selectedLibraryId ? { ...library, source_roots: rooted.source_roots } : library
          )
        )
        const preview = requireData(await client.scanLibrary(selectedLibraryId, true), '素材目录预扫描失败')
        setScanPreview(preview)

        return preview
      } finally {
        setManagementBusy(false)
      }
    },
    [client, selectedLibraryId]
  )

  const confirmScan = useCallback(async () => {
    if (!selectedLibraryId) {throw new Error('请先选择资产库')}
    setScanBusy(true)

    try {
      const result = requireData(await client.scanLibrary(selectedLibraryId, false), '资产库扫描失败')
      setScanPreview(null)
      await refreshSelectedLibrary()

      return result
    } finally {
      setScanBusy(false)
    }
  }, [client, refreshSelectedLibrary, selectedLibraryId])

  const previewScan = useCallback(async () => {
    if (!selectedLibraryId) {throw new Error('请先选择资产库')}
    setScanBusy(true)

    try {
      const preview = requireData(await client.scanLibrary(selectedLibraryId, true), '资产库预扫描失败')
      setScanPreview(preview)

      return preview
    } finally {
      setScanBusy(false)
    }
  }, [client, selectedLibraryId])

  const migrateLegacyLibrary = useCallback(async () => {
    if (!selectedLibraryId) {throw new Error('请先选择资产库')}
    setManagementBusy(true)

    try {
      const result = requireData(await client.migrateLegacyLibrary(selectedLibraryId), '旧素材库迁移失败')
      setMigrationResult(result)
      await refreshSelectedLibrary()

      return result
    } finally {
      setManagementBusy(false)
    }
  }, [client, refreshSelectedLibrary, selectedLibraryId])

  const createTimeline = useCallback(
    async (aspect: VideoAspect) => {
      if (!selectedLibraryId) {throw new Error('请先选择资产库')}
      const confirmedSegments = segments.filter(segment => matches.confirmedBySegment[segment.id])

      if (confirmedSegments.length === 0) {throw new Error('请先人工确认至少一个镜头')}
      const clipIds = confirmedSegments.map(segment => matches.confirmedBySegment[segment.id])
      setTimelineBusy(true)

      try {
        const response = await client.createTimeline(
          selectedLibraryId,
          clipIds,
          aspect,
          confirmedSegments.map(segment => ({ id: segment.id, text: segment.text }))
        )

        const result = requireData(response, '素材时间线创建失败')
        setTimeline(result)

        return result
      } finally {
        setTimelineBusy(false)
      }
    },
    [client, matches.confirmedBySegment, segments, selectedLibraryId]
  )

  const createAutomaticTimeline = useCallback(
    async (aspect: VideoAspect) => {
      if (!selectedLibraryId) {
        throw new Error('请先选择资产库')
      }

      if (segments.length === 0) {
        throw new Error('请先填写视频文案')
      }

      setMatchingAll(true)
      setTimelineBusy(true)
      setError('')

      try {
        const candidatesBySegment: Record<string, VideoLibraryClip[]> = {}

        for (const segment of segments) {
          const query = [segment.text, terms.trim()].filter(Boolean).join(' ')

          let candidates = requireData(
            await client.listClips(selectedLibraryId, { limit: AUTO_MATCH_CANDIDATE_LIMIT, query }),
            '镜头匹配失败'
          ).clips

          if (candidates.length === 0) {
            candidates = requireData(
              await client.listClips(selectedLibraryId, { limit: AUTO_MATCH_CANDIDATE_LIMIT }),
              '镜头回退匹配失败'
            ).clips
          }

          if (candidates.length === 0) {
            throw new Error('当前素材库没有可用镜头')
          }

          candidatesBySegment[segment.id] = candidates
        }

        const pool = planAutomaticClipPool(segments, candidatesBySegment)
        const selectedBySegment = Object.fromEntries(
          pool.filter(item => item.round === 0).map(item => [item.segment.id, item.clip.id])
        )

        if (Object.keys(selectedBySegment).length !== segments.length) {
          throw new Error('AI 无法为全部文案匹配镜头')
        }

        setMatches({
          candidatesBySegment,
          confirmedBySegment: selectedBySegment,
          errorsBySegment: {}
        })

        const result = requireData(
          await client.createTimeline(
            selectedLibraryId,
            pool.map(item => item.clip.id),
            aspect,
            pool.map(item => ({ id: item.segment.id, text: item.segment.text }))
          ),
          '素材时间线创建失败'
        )

        setTimeline(result)

        return result
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason))
        throw reason
      } finally {
        setMatchingAll(false)
        setTimelineBusy(false)
      }
    },
    [client, segments, selectedLibraryId, terms]
  )

  return {
    addSourceRoot,
    assets,
    clips,
    confirmClip,
    confirmScan,
    createAutomaticTimeline,
    createTimeline,
    error,
    importFiles,
    libraries,
    loadingLibraries,
    loadingLibrary,
    managementBusy,
    matchAll,
    matchSegment,
    matches,
    matchingAll,
    matchingSegmentId,
    migrateLegacyLibrary,
    migrationResult,
    previewScan,
    refreshSelectedLibrary,
    scanBusy,
    scanPreview,
    scanSelectedLibrary,
    segments,
    selectLibrary,
    selectedLibraryId,
    status,
    timeline,
    timelineBusy
  }
}
