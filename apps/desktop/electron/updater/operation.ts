import type { UpdaterApplyResultWire, UpdaterStrategy } from './index'

/** Preserve one native selection across check/apply and one owner through handoff. */
export class UpdateOperation {
  private strategy: Promise<UpdaterStrategy | null> | undefined
  private applying: boolean = false

  constructor(private readonly initialize: () => Promise<UpdaterStrategy | null>) {}

  resolve(): Promise<UpdaterStrategy | null> {
    this.strategy ??= this.initialize().catch((error: Error): never => {
      this.strategy = undefined
      throw error
    })

    return this.strategy
  }

  async apply(run: () => Promise<UpdaterApplyResultWire>): Promise<UpdaterApplyResultWire> {
    if (this.applying) {
      throw new Error('An update is already in progress.')
    }

    this.applying = true
    let handedOff: boolean = false

    try {
      const result: UpdaterApplyResultWire = await run()
      handedOff = result.handedOff === true

      return result
    } finally {
      if (!handedOff) {
        this.applying = false
      }
    }
  }
}
