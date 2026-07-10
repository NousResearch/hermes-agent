import type { ComponentType } from 'react'

import { Archive, AudioLines, FileImage, FileText, Globe, Play } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { ArtifactRecord } from './artifact-utils'

type PreviewKind =
  | 'archive'
  | 'audio'
  | 'code'
  | 'data'
  | 'document'
  | 'file'
  | 'image'
  | 'presentation'
  | 'spreadsheet'
  | 'video'
  | 'web'

interface ArtifactPreviewDescriptor {
  accentClass: string
  badge: string
  icon: ComponentType<{ className?: string }>
  kind: PreviewKind
  surfaceClass: string
}

const ARCHIVE_EXTENSIONS = new Set(['7z', 'gz', 'rar', 'tar', 'tgz', 'zip'])
const AUDIO_EXTENSIONS = new Set(['flac', 'm4a', 'mp3', 'ogg', 'opus', 'wav'])

const CODE_EXTENSIONS = new Set([
  'bash',
  'c',
  'cc',
  'cpp',
  'css',
  'go',
  'h',
  'hpp',
  'html',
  'ini',
  'java',
  'js',
  'jsx',
  'py',
  'rb',
  'rs',
  'sass',
  'scss',
  'sh',
  'toml',
  'ts',
  'tsx',
  'yaml',
  'yml',
  'zsh'
])

const DATA_EXTENSIONS = new Set(['csv', 'json'])
const DOCUMENT_EXTENSIONS = new Set(['doc', 'docx', 'md', 'pdf', 'txt'])
const IMAGE_EXTENSIONS = new Set(['avif', 'bmp', 'gif', 'jpeg', 'jpg', 'png', 'svg', 'webp'])
const PRESENTATION_EXTENSIONS = new Set(['ppt', 'pptx'])
const SPREADSHEET_EXTENSIONS = new Set(['xls', 'xlsx'])
const VIDEO_EXTENSIONS = new Set(['avi', 'mkv', 'mov', 'mp4', 'webm'])

function extensionFromValue(value: string): string {
  const withoutQuery = value.split(/[?#]/, 1)[0] || ''
  const filename = withoutQuery.split(/[\\/]/).filter(Boolean).pop()?.toLowerCase() || ''

  return filename.includes('.') ? filename.split('.').pop() || '' : ''
}

function badgeFor(extension: string, fallback: string): string {
  return (extension || fallback).slice(0, 6).toUpperCase()
}

export function artifactPreviewDescriptor(artifact: Pick<ArtifactRecord, 'kind' | 'value'>): ArtifactPreviewDescriptor {
  const extension = extensionFromValue(artifact.value)

  if (artifact.kind === 'image' || IMAGE_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-fuchsia-300',
      badge: badgeFor(extension, 'IMG'),
      icon: FileImage,
      kind: 'image',
      surfaceClass: 'from-fuchsia-500/18 via-violet-500/8 to-transparent'
    }
  }

  if (ARCHIVE_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-amber-300',
      badge: badgeFor(extension, 'ARC'),
      icon: Archive,
      kind: 'archive',
      surfaceClass: 'from-amber-500/18 via-orange-500/8 to-transparent'
    }
  }

  if (AUDIO_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-cyan-300',
      badge: badgeFor(extension, 'AUD'),
      icon: AudioLines,
      kind: 'audio',
      surfaceClass: 'from-cyan-500/18 via-sky-500/8 to-transparent'
    }
  }

  if (VIDEO_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-rose-300',
      badge: badgeFor(extension, 'VID'),
      icon: Play,
      kind: 'video',
      surfaceClass: 'from-rose-500/18 via-red-500/8 to-transparent'
    }
  }

  if (CODE_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-violet-300',
      badge: badgeFor(extension, 'CODE'),
      icon: FileText,
      kind: 'code',
      surfaceClass: 'from-violet-500/18 via-indigo-500/8 to-transparent'
    }
  }

  if (DATA_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-emerald-300',
      badge: badgeFor(extension, 'DATA'),
      icon: FileText,
      kind: 'data',
      surfaceClass: 'from-emerald-500/18 via-teal-500/8 to-transparent'
    }
  }

  if (PRESENTATION_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-orange-300',
      badge: badgeFor(extension, 'PRES'),
      icon: FileText,
      kind: 'presentation',
      surfaceClass: 'from-orange-500/18 via-amber-500/8 to-transparent'
    }
  }

  if (SPREADSHEET_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-lime-300',
      badge: badgeFor(extension, 'SHEET'),
      icon: FileText,
      kind: 'spreadsheet',
      surfaceClass: 'from-lime-500/18 via-emerald-500/8 to-transparent'
    }
  }

  if (DOCUMENT_EXTENSIONS.has(extension)) {
    return {
      accentClass: 'text-blue-300',
      badge: badgeFor(extension, 'DOC'),
      icon: FileText,
      kind: 'document',
      surfaceClass: 'from-blue-500/18 via-sky-500/8 to-transparent'
    }
  }

  if (artifact.kind === 'link') {
    return {
      accentClass: 'text-sky-300',
      badge: 'WEB',
      icon: Globe,
      kind: 'web',
      surfaceClass: 'from-sky-500/18 via-cyan-500/8 to-transparent'
    }
  }

  return {
    accentClass: 'text-slate-300',
    badge: badgeFor(extension, 'FILE'),
    icon: FileText,
    kind: 'file',
    surfaceClass: 'from-slate-500/18 via-zinc-500/8 to-transparent'
  }
}

interface ArtifactPreviewThumbnailProps {
  ariaLabel: string
  artifact: ArtifactRecord
  className?: string
}

export function ArtifactPreviewThumbnail({ ariaLabel, artifact, className }: ArtifactPreviewThumbnailProps) {
  const descriptor = artifactPreviewDescriptor(artifact)
  const Icon = descriptor.icon

  return (
    <div
      aria-label={ariaLabel}
      className={cn(
        `relative flex h-12 w-20 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/8 bg-linear-to-br ${descriptor.surfaceClass}`,
        className
      )}
      role="img"
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_75%_20%,rgba(255,255,255,0.13),transparent_38%)]" />
      <Icon className={`relative h-5 w-5 ${descriptor.accentClass}`} />
      <span className="absolute right-1 bottom-0.5 max-w-[4.5rem] truncate rounded-sm bg-black/25 px-1 py-0.5 font-mono text-[8px] font-semibold tracking-[0.08em] text-white/85">
        {descriptor.badge}
      </span>
    </div>
  )
}
