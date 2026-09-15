// Minimal WebGPU renderer for the vendored orb shader.
//
// Ported from https://github.com/LerSent001/orb (src/orb-renderer.ts).
// Copyright (c) LerSent001. Licensed under the MIT License.
// Adapted for hermes-agent: the editor's thinking/idle state-transition
// controller is dropped — the desktop orb always renders one param set —
// and errors are reported in English via `onError` so the caller can fall
// back to the standard indicator.

import { type OrbParams, styleFlowIndexes } from './orb-params'
import { orbShaderSource } from './orb-shader'
import { orbUniformFloatCount, writeOrbUniforms } from './orb-uniforms'

const particleRibbonInstanceCount = 384 * 96 * 6

// WebGPU usage flags. TypeScript's DOM lib types these as plain numbers, so
// the flag constants are declared here with their WebGPU spec values.
const GPUBufferUsageFlags = { UNIFORM: 0x40, COPY_DST: 0x08 } as const
const GPUTextureUsageFlags = { RENDER_ATTACHMENT: 0x10, TEXTURE_BINDING: 0x04 } as const

export interface OrbRendererOptions {
  canvas: HTMLCanvasElement
  /** Read live so a settings preview can retarget the orb without remounting. */
  getParams: () => OrbParams
  onError: (error: Error) => void
  onReady?: () => void
}

/** Synchronous capability probe — true when the browser exposes WebGPU. */
export function isOrbWebGpuSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu != null
}

export function createOrbRenderer({ canvas, getParams, onError, onReady }: OrbRendererOptions): () => void {
  let disposed = false
  let animationFrame = 0
  let device: GPUDevice | null = null
  let ribbonTarget: GPUTexture | null = null
  let readyNotified = false
  let failed = false
  let lastFrameAt: number | null = null
  let motionPhase = 0

  function fail(error: Error): void {
    if (disposed || failed) {return}
    failed = true
    cancelAnimationFrame(animationFrame)
    ribbonTarget?.destroy()
    device?.destroy()
    onError(error)
  }

  async function start(): Promise<void> {
    if (!isOrbWebGpuSupported()) {
      throw new Error('WebGPU is not available in this browser')
    }

    const adapter = await navigator.gpu!.requestAdapter()

    if (!adapter) {
      throw new Error('No WebGPU adapter found')
    }

    device = await adapter.requestDevice()

    if (disposed) {
      device.destroy()

      return
    }

    const context = canvas.getContext('webgpu')

    if (!context) {
      throw new Error('Could not create a WebGPU canvas context')
    }

    const gpuContext = context as GPUCanvasContext
    const format = navigator.gpu!.getPreferredCanvasFormat()
    gpuContext.configure({ device, format, alphaMode: 'premultiplied' })

    const shader = device.createShaderModule({ label: 'orb-glass-liquid', code: orbShaderSource })
    const compilation = await shader.getCompilationInfo()
    const compilationErrors = compilation.messages.filter(message => message.type === 'error')

    if (compilationErrors.length > 0) {
      throw new Error(
        compilationErrors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')
      )
    }

    const pipeline = device.createRenderPipeline({
      label: 'orb-glass-liquid-pipeline',
      layout: 'auto',
      vertex: { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module: shader,
        entryPoint: 'fs_main',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    const ribbonPipeline = device.createRenderPipeline({
      label: 'particle-ribbon-pipeline',
      layout: 'auto',
      vertex: { module: shader, entryPoint: 'ribbon_vs_main' },
      fragment: {
        module: shader,
        entryPoint: 'ribbon_fs_main',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    const ribbonCompositePipeline = device.createRenderPipeline({
      label: 'particle-ribbon-glass-composite-pipeline',
      layout: 'auto',
      vertex: { module: shader, entryPoint: 'vs_main' },
      fragment: {
        module: shader,
        entryPoint: 'ribbon_composite_fs_main',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    const values = new Float32Array(orbUniformFloatCount)

    const uniformBuffer = device.createBuffer({
      size: values.byteLength,
      usage: GPUBufferUsageFlags.UNIFORM | GPUBufferUsageFlags.COPY_DST
    })

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    })

    const ribbonBindGroup = device.createBindGroup({
      layout: ribbonPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    })

    const ribbonSampler = device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear'
    })

    let ribbonCompositeBindGroup: GPUBindGroup | null = null

    device.lost.then(info => {
      fail(new Error(`WebGPU device lost: ${info.message || info.reason}`))
    })

    device.addEventListener('uncapturederror', event => {
      event.preventDefault()
      fail(new Error(`WebGPU render error: ${(event as GPUUncapturedErrorEvent).error.message}`))
    })

    function resize(): void {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.max(1, Math.floor(canvas.clientWidth * dpr))
      const height = Math.max(1, Math.floor(canvas.clientHeight * dpr))

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        ribbonTarget?.destroy()
        ribbonTarget = null
        ribbonCompositeBindGroup = null
      }
    }

    function ensureRibbonTarget(): void {
      if (ribbonTarget && ribbonCompositeBindGroup) {return}
      ribbonTarget = device!.createTexture({
        label: 'particle-ribbon-offscreen-texture',
        size: { width: canvas.width, height: canvas.height },
        format,
        usage: GPUTextureUsageFlags.RENDER_ATTACHMENT | GPUTextureUsageFlags.TEXTURE_BINDING
      })
      ribbonCompositeBindGroup = device!.createBindGroup({
        layout: ribbonCompositePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: ribbonTarget.createView() },
          { binding: 2, resource: ribbonSampler }
        ]
      })
    }

    function frame(now: number): void {
      if (disposed || failed || !device) {
        return
      }

      try {
        resize()

        const params = getParams()
        const frameDelta = lastFrameAt === null ? 0 : Math.min(0.1, Math.max(0, (now - lastFrameAt) / 1000))
        lastFrameAt = now
        motionPhase += frameDelta * Math.max(params.speed, 0)
        const shaderTime = motionPhase / Math.max(params.speed, 0.001)

        writeOrbUniforms(values, canvas.width, canvas.height, shaderTime, params)
        device.queue.writeBuffer(uniformBuffer, 0, values)

        const isParticleRibbon = styleFlowIndexes[params.style] === styleFlowIndexes.particleRibbon
        const encoder = device.createCommandEncoder()

        if (isParticleRibbon) {
          ensureRibbonTarget()

          const particlePass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: ribbonTarget!.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store'
              }
            ]
          })

          particlePass.setPipeline(ribbonPipeline)
          particlePass.setBindGroup(0, ribbonBindGroup)
          particlePass.draw(6, particleRibbonInstanceCount, 0, 0)
          particlePass.end()
        }

        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: gpuContext.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear',
              storeOp: 'store'
            }
          ]
        })

        if (isParticleRibbon) {
          pass.setPipeline(ribbonCompositePipeline)
          pass.setBindGroup(0, ribbonCompositeBindGroup!)
        } else {
          pass.setPipeline(pipeline)
          pass.setBindGroup(0, bindGroup)
        }

        pass.draw(3, 1, 0, 0)
        pass.end()
        device.queue.submit([encoder.finish()])

        if (!readyNotified) {
          readyNotified = true
          onReady?.()
        }

        animationFrame = requestAnimationFrame(frame)
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)))
      }
    }

    animationFrame = requestAnimationFrame(frame)
  }

  start().catch((error: unknown) => {
    fail(error instanceof Error ? error : new Error(String(error)))
  })

  return () => {
    disposed = true
    cancelAnimationFrame(animationFrame)
    ribbonTarget?.destroy()
    device?.destroy()
  }
}
