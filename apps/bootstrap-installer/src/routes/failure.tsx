import { useStore } from '@nanostores/react'
import { FileText, RefreshCw } from 'lucide-react'
import { type CSSProperties } from 'react'

import { Button } from '../components/button'
import { $logPath, type BootstrapStateModel, openLogDir, startInstall } from '../store'

interface FailureProps {
  bootstrap: BootstrapStateModel
}

/*
 * Failure screen. Same hero treatment as Welcome/Success — the wordmark
 * carries the brand across every terminal state. The error text (usually
 * the tail of bootstrap-north-forge.ps1's own output) sits below in muted
 * text; Retry re-runs the bootstrap, Open logs opens the log folder.
 */
export default function Failure({ bootstrap }: FailureProps) {
  const logPath = useStore($logPath)

  return (
    <div className="nf-fade-in flex h-full flex-col items-center justify-center gap-6 px-12 py-10">
      <div className="w-full max-w-2xl min-w-0 text-center">
        <p
          className="fit-text mx-auto mb-4 w-full font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-destructive mix-blend-plus-lighter dark:text-destructive/90"
          style={
            {
              '--fit-text-line-height': '0.9',
              '--fit-text-max': '5rem',
              '--fit-text-min': '2.25rem'
            } as CSSProperties
          }
        >
          <span>
            <span>Setup didn&rsquo;t finish</span>
          </span>
          <span aria-hidden="true">Setup didn&rsquo;t finish</span>
        </p>

        <pre className="m-0 mx-auto max-w-xl overflow-x-auto whitespace-pre-wrap text-left text-xs leading-normal tracking-tight text-muted-foreground">
          {bootstrap.error ?? 'Something went wrong while running bootstrap-north-forge.ps1.'}
        </pre>
      </div>

      <div className="flex items-center gap-3">
        <Button className="gap-1.5" onClick={() => void startInstall()}>
          <RefreshCw />
          Retry
        </Button>
        <Button className="gap-1.5" onClick={() => void openLogDir()} variant="text">
          <FileText />
          Open logs
        </Button>
      </div>

      {logPath && (
        <p className="max-w-lg text-center text-xs text-muted-foreground/70">
          Log: <code className="font-mono">{logPath}</code>
        </p>
      )}
    </div>
  )
}
