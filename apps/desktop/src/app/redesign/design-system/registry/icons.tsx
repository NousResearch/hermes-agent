import { AgentMark } from '@/components/ui/agent-mark'
import { AvatarChip } from '@/components/ui/avatar-chip'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { FileTypeIcon } from '@/components/ui/file-type-icon'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { ToolIcon } from '@/components/ui/tool-icon'
import { PIXEL_ICONS, type PixelIconName } from '@/lib/pixel-icon-registry'
import { profileColor } from '@/lib/profile-color'

import type { GalleryEntry } from './types'

const PIXEL_NAMES = Object.keys(PIXEL_ICONS) as PixelIconName[]

export const ICONS: readonly GalleryEntry[] = [
  {
    title: 'PixelIcon',
    source: 'components/ui/pixel-icon.tsx',
    use: "The redesign's icon set; every name in lib/pixel-icon-registry.ts.",
    group: 'icons',
    dense: true,
    examples: PIXEL_NAMES.map(name => ({ label: name, render: () => <PixelIcon className="ds-icon" name={name} /> }))
  },
  {
    title: 'Codicon',
    source: 'components/ui/codicon.tsx',
    use: 'An icon by its codicon name; in the redesign it draws the pixel icon mapped to that name.',
    group: 'icons',
    examples: ['add', 'search', 'settings-gear', 'ellipsis', 'loading'].map(name => ({
      label: name,
      render: () => <Codicon name={name} spinning={name === 'loading'} />
    }))
  },
  {
    title: 'DisclosureCaret',
    source: 'components/ui/disclosure-caret.tsx',
    use: 'The caret of a section that opens and closes.',
    group: 'icons',
    examples: [
      { label: 'closed', render: () => <DisclosureCaret open={false} /> },
      { label: 'open', render: () => <DisclosureCaret open /> }
    ]
  },
  {
    title: 'FileTypeIcon',
    source: 'components/ui/file-type-icon.tsx',
    use: 'A file by its name or language.',
    group: 'icons',
    examples: ['notes.md', 'main.py', 'index.tsx', 'data.json'].map(path => ({
      label: path,
      render: () => <FileTypeIcon path={path} />
    }))
  },
  {
    title: 'ToolIcon',
    source: 'components/ui/tool-icon.tsx',
    use: 'The glyph of a tool call in the transcript.',
    group: 'icons',
    examples: ['edit', 'eye', 'file', 'diff'].map(name => ({ label: name, render: () => <ToolIcon name={name} /> }))
  },
  {
    title: 'AgentMark',
    source: 'components/ui/agent-mark.tsx',
    use: "An agent's tile in its own colour (FigmaAgentIcon underneath).",
    group: 'icons',
    covers: ['components/ui/figma-agent-icon.tsx'],
    examples: [
      { label: 'research', render: () => <AgentMark fallback={null} profile="research" /> },
      { label: 'size 32', render: () => <AgentMark fallback={null} profile="research" size={32} /> },
      ...(['subtle', 'outline'] as const).map(variant => ({
        label: `FigmaAgentIcon ${variant}`,
        render: () => <FigmaAgentIcon color={profileColor('research')} size={32} variant={variant} />
      }))
    ]
  },
  {
    title: 'ProfileGlyph',
    source: 'components/ui/profile-glyph.tsx',
    use: "A profile's initial on a tint of its colour; the default profile is home.",
    group: 'icons',
    examples: [
      {
        label: 'named',
        render: () => <ProfileGlyph color={profileColor('research')} isDefault={false} name="research" />
      },
      { label: 'default', render: () => <ProfileGlyph color={null} isDefault name="default" /> }
    ]
  },
  {
    title: 'AvatarChip',
    source: 'components/ui/avatar-chip.tsx',
    use: 'A small square mark for a person, a provider or a service, with a monogram when it has no logo.',
    group: 'icons',
    examples: [{ label: 'monogram', render: () => <AvatarChip name="Research" /> }]
  }
]
