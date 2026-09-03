import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { LUNAR_CITY_ASSET_MANIFEST } from './world-assets'

describe('Lunar City asset manifest', () => {
  it('covers the full interactive world asset contract', () => {
    expect(LUNAR_CITY_ASSET_MANIFEST.schemaVersion).toBe(2)
    expect(LUNAR_CITY_ASSET_MANIFEST.glb).toBe('lunar-city/lunar-city-baseline.glb')
    expect(LUNAR_CITY_ASSET_MANIFEST.generated3dBoardGlb).toBe(
      'lunar-city/generated-3d/lunar-city-generated-assets-board.glb'
    )
    expect(LUNAR_CITY_ASSET_MANIFEST.generated3dBoardPreview).toBe(
      'lunar-city/generated-3d/lunar-city-generated-assets-board.png'
    )
    expect(LUNAR_CITY_ASSET_MANIFEST.generated3dManifest).toBe(
      'lunar-city/generated-3d/generated-assets-metadata.json'
    )
    expect(LUNAR_CITY_ASSET_MANIFEST.productionAssetPipeline).toEqual({
      highPolyMasterFirst: true,
      productionSource: 'full_resolution_high_poly_master_assets',
      rejectedSources: ['raw_scene_crop_image_to_3d'],
      retopology: 'derive_smart_low_poly_lods_from_master',
      textureBake: 'bake_2k_default_4k_hero_pbr_from_master'
    })
    expect(LUNAR_CITY_ASSET_MANIFEST.heroAssetGlb).toBe('lunar-city/hero-assets/lunar-city-hero-assets.glb')
    expect(LUNAR_CITY_ASSET_MANIFEST.heroAssetManifest).toBe('lunar-city/hero-assets/hero-assets-manifest.json')
    expect(LUNAR_CITY_ASSET_MANIFEST.heroAssetPreview).toBe('lunar-city/hero-assets/lunar-city-hero-assets.png')
    expect(LUNAR_CITY_ASSET_MANIFEST.masterAssetManifest).toBe('lunar-city/master-assets/master-asset-manifest.json')
    expect(LUNAR_CITY_ASSET_MANIFEST.profileManifest).toBe('lunar-city/profile-assets.json')
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'building')).toHaveLength(8)
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'character')).toHaveLength(19)
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.some(asset => asset.id === 'terrain-colony-basin')).toBe(true)
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.some(asset => asset.id === 'road-network-primary')).toBe(true)
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.some(asset => asset.id === 'dispatcher-cube')).toBe(true)
    expect(LUNAR_CITY_ASSET_MANIFEST.animationClips.map(clip => clip.clip)).toEqual([
      'idle',
      'walk',
      'work',
      'carry',
      'inspect',
      'repair',
      'talk',
      'wait',
      'panic',
      'celebrate',
      'rest',
      'return'
    ])
    expect(LUNAR_CITY_ASSET_MANIFEST.validation.requires).toContain('buildings_do_not_overlap')
  })

  it('maps every role visual identity to a home building or scene object', () => {
    const buildings = new Set(
      LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'building' || asset.kind === 'prop').map(asset => asset.id)
    )

    for (const asset of LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'character')) {
      for (const binding of asset.bindings ?? []) {
        if (binding.homeBuilding) {
          expect(buildings.has(binding.homeBuilding)).toBe(true)
        }
      }
    }
  })

  it('declares texture slots only at approved 2k or 4k resolutions', () => {
    expect(LUNAR_CITY_ASSET_MANIFEST.textures.length).toBeGreaterThan(8)

    for (const texture of LUNAR_CITY_ASSET_MANIFEST.textures) {
      expect(['2k', '4k']).toContain(texture.maxResolution)

      for (const slot of texture.slots) {
        expect(slot.uri).toContain(`lunar-city/textures/${texture.id}/`)
        expect(['2k', '4k']).toContain(slot.resolution)
      }
    }
  })

  it('tracks the generated sculpted hero asset library', () => {
    const manifest = JSON.parse(
      readFileSync(join(process.cwd(), 'public/lunar-city/hero-assets/hero-assets-manifest.json'), 'utf8')
    ) as {
      assetCount: number
      assetQuality: Array<{
        animationRigWireCount: number
        collection: string
        heroComponentCount: number
        id: string
        lodPolicy: string[]
        proceduralPbrMaterialCount: number
        retopologyTarget: string
        sculptedSurfaceCount: number
      }>
      buildingPreview: string
      buildingDetailComponentCount: number
      buildings: Array<{ collection: string; id: string }>
      characterPreview: string
      children: Array<{ collection: string; id: string }>
      heroMeshComponentCount: number
      leaderPreview: string
      leaders: Array<{ collection: string; id: string; signature: string }>
      lods: Array<{ id: string; levels: Record<string, unknown>; sourceCollection: string }>
      proceduralPbrMaterialCount: number
      proceduralPbrMaterials: string[]
      sculptedCharacterCoreComponentCount: number
      sculptedCharacterLimbComponentCount: number
      sculptedSurfaceComponentCount: number
      uniqueLeaderSignatureCount: number
      validation: Record<string, boolean>
      workers: Array<{ collection: string; id: string }>
    }

    expect(manifest.assetCount).toBe(26)
    expect(manifest.buildingPreview).toBe('lunar-city/hero-assets/lunar-city-hero-buildings.png')
    expect(manifest.characterPreview).toBe('lunar-city/hero-assets/lunar-city-hero-characters.png')
    expect(manifest.leaderPreview).toBe('lunar-city/hero-assets/lunar-city-hero-leaders.png')
    expect(manifest.buildings).toHaveLength(8)
    expect(manifest.leaders).toHaveLength(8)
    expect(new Set(manifest.leaders.map(leader => leader.signature)).size).toBe(8)
    expect(manifest.workers).toHaveLength(6)
    expect(manifest.children).toHaveLength(4)
    expect(manifest.heroMeshComponentCount).toBeGreaterThan(600)
    expect(manifest.buildingDetailComponentCount).toBeGreaterThanOrEqual(160)
    expect(manifest.uniqueLeaderSignatureCount).toBeGreaterThanOrEqual(24)
    expect(manifest.sculptedSurfaceComponentCount).toBeGreaterThanOrEqual(148)
    expect(manifest.sculptedCharacterCoreComponentCount).toBeGreaterThanOrEqual(36)
    expect(manifest.sculptedCharacterLimbComponentCount).toBeGreaterThanOrEqual(72)
    expect(manifest.proceduralPbrMaterialCount).toBeGreaterThanOrEqual(12)
    expect(manifest.proceduralPbrMaterials).toContain('Hero white hull PBR')
    expect(manifest.proceduralPbrMaterials).toContain('Hero leader fur')
    expect(manifest.validation.usesContinuousSculptedSurfaces).toBe(true)
    expect(manifest.validation.usesContinuousCharacterCoreMeshes).toBe(true)
    expect(manifest.validation.usesContinuousCharacterLimbMeshes).toBe(true)
    expect(manifest.validation.usesDetailedBuildingFacades).toBe(true)
    expect(manifest.validation.usesUniqueLeaderSignatures).toBe(true)
    expect(manifest.validation.usesProceduralPbrMaterials).toBe(true)
    expect(manifest.validation.freeLocalGenerationOnly).toBe(true)
    expect(manifest.validation.noRawSoulContent).toBe(true)
    expect(manifest.validation.tracksPerAssetQuality).toBe(true)
    expect(manifest.validation.tracksLodBudgets).toBe(true)
    expect(manifest.assetQuality).toHaveLength(26)
    expect(manifest.lods).toHaveLength(26)
    for (const asset of [...manifest.buildings, ...manifest.leaders, ...manifest.workers, ...manifest.children]) {
      expect(asset.collection).toBe(`Hero Asset - ${asset.id}`)
      const quality = manifest.assetQuality.find(entry => entry.id === asset.id)
      const lod = manifest.lods.find(entry => entry.id === asset.id)
      expect(quality).toBeTruthy()
      expect(lod).toBeTruthy()
      expect(quality?.collection).toBe(asset.collection)
      expect(lod?.sourceCollection).toBe(asset.collection)
      expect(quality?.heroComponentCount).toBeGreaterThan(0)
      expect(quality?.proceduralPbrMaterialCount).toBeGreaterThan(0)
      expect(quality?.lodPolicy).toEqual(['hero', 'high', 'medium', 'low'])
      expect(quality?.retopologyTarget).toBe('quad_dominant_smart_low_poly')
      expect(Object.keys(lod?.levels ?? {}).sort()).toEqual(['hero', 'high', 'low', 'medium'])
    }
  })

  it('tracks local image-to-3D meshes generated from the approved reference crops', () => {
    const referenceManifest = JSON.parse(
      readFileSync(join(process.cwd(), 'public/lunar-city/generated-3d/reference-crops/reference-crops-manifest.json'), 'utf8')
    ) as {
      cards: Array<{ id: string; kind: string; targetMesh: string; uri: string }>
      privacy: {
        containsPrivateProfileIdentifiers: boolean
        usesRawSoulContent: boolean
      }
    }
    const generatedManifest = JSON.parse(
      readFileSync(join(process.cwd(), 'public/lunar-city/generated-3d/generated-assets-metadata.json'), 'utf8')
    ) as {
      assetCount: number
      assets: Array<{ id: string; kind: string; mesh: string; pbrStatus: string; sourceReferenceCrop: string; status: string }>
      importedCount: number
      missingCount: number
      privacy: {
        containsPrivateProfileIdentifiers: boolean
        usesRawSoulContent: boolean
      }
      productionEligibility: string
      productionUse: string
      rejectionReason: string
    }

    expect(referenceManifest.cards).toHaveLength(23)
    expect(referenceManifest.privacy.usesRawSoulContent).toBe(false)
    expect(referenceManifest.privacy.containsPrivateProfileIdentifiers).toBe(false)
    expect(generatedManifest.assetCount).toBe(23)
    expect(generatedManifest.importedCount).toBe(23)
    expect(generatedManifest.missingCount).toBe(0)
    expect(generatedManifest.productionUse).toBe('reference_only')
    expect(generatedManifest.productionEligibility).toBe('rejected_for_production')
    expect(generatedManifest.rejectionReason).toContain('full-resolution high-poly master assets')
    expect(generatedManifest.privacy.usesRawSoulContent).toBe(false)
    expect(generatedManifest.privacy.containsPrivateProfileIdentifiers).toBe(false)

    const expectedIds = [
      'building-library',
      'building-research-lab',
      'building-arts-studio',
      'building-engineering',
      'building-operations-depot',
      'building-release-gatehouse',
      'building-triage-clinic',
      'building-council-hall',
      'building-review-office',
      'building-archive',
      'leader-owl-archivist',
      'leader-fox-scientist',
      'leader-raccoon-artist',
      'leader-eagle-councillor',
      'leader-badger-engineer',
      'leader-hawk-reviewer',
      'leader-owl-historian',
      'worker-bot-round',
      'worker-bot-carrying',
      'worker-bot-review',
      'child-bot-garden',
      'vehicle-bus',
      'prop-break-garden'
    ]
    expect(referenceManifest.cards.map(card => card.id)).toEqual(expectedIds)
    expect(generatedManifest.assets.map(asset => asset.id)).toEqual(expectedIds)

    for (const asset of generatedManifest.assets) {
      expect(asset.status).toBe('imported')
      expect(['source_materials', 'needs_rebake']).toContain(asset.pbrStatus)
      expect(asset.mesh).toBe(`lunar-city/generated-3d/meshes/${asset.id}.glb`)
      expect(asset.sourceReferenceCrop).toBe(`lunar-city/generated-3d/reference-crops/${asset.id}.png`)
      expect(existsSync(join(process.cwd(), 'public', asset.mesh))).toBe(true)
      expect(existsSync(join(process.cwd(), 'public', asset.sourceReferenceCrop))).toBe(true)
    }

    expect(existsSync(join(process.cwd(), 'public/lunar-city/generated-3d/lunar-city-generated-assets-board.blend'))).toBe(true)
    expect(existsSync(join(process.cwd(), 'public/lunar-city/generated-3d/lunar-city-generated-assets-board.glb'))).toBe(true)
    expect(existsSync(join(process.cwd(), 'public/lunar-city/generated-3d/lunar-city-generated-assets-board.png'))).toBe(true)
  })

  it('fails closed until production high-poly master assets exist', () => {
    const manifest = JSON.parse(
      readFileSync(join(process.cwd(), 'public/lunar-city/master-assets/master-asset-manifest.json'), 'utf8')
    ) as {
      counts: { missing: number; present: number; required: number }
      pipeline: {
        animation: string
        retopology: string
        source: string
        textureBake: string
      }
      productionReady: boolean
      productionUse: string
      rejectedProductionSources: string[]
      requiredAssets: Array<{
        acceptance: {
          minimumTriangleCount: number
          rejectIf: string[]
          requiresLods: string[]
          sourceQuality: string
        }
        heroAsset: boolean
        id: string
        selectedSource: string | null
        status: string
      }>
      validation: Record<string, boolean>
    }

    expect(manifest.productionUse).toBe('production_source_intake')
    expect(manifest.productionReady).toBe(false)
    expect(manifest.pipeline.source).toBe('full_resolution_high_poly_master_assets')
    expect(manifest.pipeline.retopology).toBe('derive_smart_low_poly_lods_from_master')
    expect(manifest.pipeline.textureBake).toBe('bake_2k_default_4k_hero_pbr_from_master')
    expect(manifest.rejectedProductionSources).toEqual([
      'raw_scene_crop_image_to_3d',
      'floating_blob_meshes',
      'simple_mascot_generator',
      'flat_reference_planes'
    ])
    expect(manifest.counts).toEqual({ required: 36, present: 0, missing: 36 })
    expect(manifest.requiredAssets).toHaveLength(36)
    expect(manifest.validation.failsClosedUntilEveryRequiredMasterExists).toBe(true)
    expect(manifest.validation.requiresNoRawSoulContent).toBe(true)
    expect(manifest.validation.requiresNoPrivateProfileIdentifiers).toBe(true)

    const ids = manifest.requiredAssets.map(asset => asset.id)
    expect(ids).toContain('leader-fox-scientist')
    expect(ids).toContain('leader-owl-archivist')
    expect(ids).toContain('worker-review')
    expect(ids).toContain('building-research-lab')
    expect(ids).toContain('building-release-gatehouse')
    expect(ids).toContain('terrain-colony-basin')
    expect(ids).toContain('skybox-lunar-orbit')

    for (const asset of manifest.requiredAssets) {
      expect(asset.status).toBe('missing')
      expect(asset.selectedSource).toBeNull()
      expect(asset.acceptance.sourceQuality).toBe('full_resolution_high_poly_master')
      expect(asset.acceptance.rejectIf).toContain('floating_blob')
      expect(asset.acceptance.rejectIf).toContain('simple_mascot_placeholder')
      expect(asset.acceptance.rejectIf).toContain('flat_billboard_or_reference_plane')
      expect(asset.acceptance.minimumTriangleCount).toBeGreaterThanOrEqual(asset.heroAsset ? 120000 : 45000)
    }
  })
})
