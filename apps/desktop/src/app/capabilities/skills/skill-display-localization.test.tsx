import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { OfficialSkillInfo, SkillInfo } from '@/types/hermes'

import { OfficialSkillDetail } from './official-skill-detail'
import { SkillDetail } from './skill-detail'
import { filteredOfficial, filteredSkills } from './skills-data'

const mocks = vi.hoisted(() => ({ getSkillContent: vi.fn(), previewSkillHub: vi.fn() }))

vi.mock('@/hermes', async () => {
  const actual = (await vi.importActual('@/hermes')) as Record<string, unknown>

  return {
    ...actual,
    getSkillContent: (name: string, profile?: unknown) => mocks.getSkillContent(name, profile),
    previewSkillHub: (identifier: string, profile?: unknown) => mocks.previewSkillHub(identifier, profile)
  }
})

// 技能标识（/pdf、mutation 用的 name）与后端 description 都必须原样保留，
// 汉化只发生在显示层。
const installedPdf: SkillInfo = {
  category: 'productivity',
  description: 'PDF files: create, read, merge, fill, OCR, edit text.',
  enabled: true,
  name: 'pdf'
}

const officialPdf: OfficialSkillInfo = {
  category: 'productivity',
  description: 'PDF files: create, read, merge, fill, OCR, edit text.',
  identifier: 'pdf',
  installed: false,
  name: 'pdf',
  tags: ['documents']
}

const SKILL_MD = '---\nname: pdf\n---\n\n# PDF\nmerge pages with qpdf\n'

function Providers({ children, locale = 'zh' }: { children: ReactNode; locale?: 'en' | 'zh' }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return (
    <QueryClientProvider client={client}>
      <I18nProvider configClient={null} initialLocale={locale}>
        {children}
      </I18nProvider>
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  mocks.getSkillContent.mockReset()
  mocks.previewSkillHub.mockReset()
})

describe('skills display localization', () => {
  it('shows a Chinese detail title that keeps the identifier, and leaves the body untranslated', async () => {
    mocks.getSkillContent.mockResolvedValue({ content: SKILL_MD, name: 'pdf', path: '/skills/pdf/SKILL.md' })

    render(
      <Providers>
        <SkillDetail onArchive={() => {}} onEdit={() => {}} skill={installedPdf} />
      </Providers>
    )

    const title = (await screen.findByText('PDF 处理 · pdf')).textContent ?? ''
    expect(title).toMatch(/\p{Script=Han}/u)
    expect(title).toContain('pdf')
    expect(screen.getByText('PDF 文件：创建、读取、合并、填写、OCR、编辑文本。')).toBeTruthy()
    // 正文是技能内容本身，逐字保持原文。
    expect(await screen.findByText(/merge pages with qpdf/)).toBeTruthy()
  })

  it('localizes the catalog pane and still previews the original identifier', async () => {
    mocks.previewSkillHub.mockResolvedValue({ skill_md: SKILL_MD })

    render(
      <Providers>
        <OfficialSkillDetail installing={false} onInstall={() => {}} skill={officialPdf} />
      </Providers>
    )

    expect(await screen.findByText('PDF 处理 · pdf')).toBeTruthy()
    await waitFor(() => expect(mocks.previewSkillHub).toHaveBeenCalledWith('pdf', undefined))
  })

  it('finds an installed skill by its Chinese description while keeping English search and object identity', () => {
    const skills = [installedPdf]

    const byChinese = filteredSkills(skills, '合并', true, 'zh')
    expect(byChinese).toHaveLength(1)
    expect(byChinese[0]).toBe(installedPdf)

    expect(filteredSkills(skills, 'merge', true, 'zh')).toEqual([installedPdf])
    expect(filteredSkills(skills, 'merge', true, 'en')).toEqual([installedPdf])
    // 英文界面不该被中文说明命中。
    expect(filteredSkills(skills, '合并', true, 'en')).toHaveLength(0)
  })

  it('finds a catalog skill by its Chinese description', () => {
    expect(filteredOfficial([officialPdf], '合并', 'zh')).toEqual([officialPdf])
    expect(filteredOfficial([officialPdf], '效率办公', 'zh')).toEqual([officialPdf])
    expect(filteredOfficial([officialPdf], '合并', 'en')).toHaveLength(0)
  })

  it('leaves the English pane on the original name and description', async () => {
    mocks.getSkillContent.mockResolvedValue({ content: SKILL_MD, name: 'pdf', path: '/skills/pdf/SKILL.md' })

    render(
      <Providers locale="en">
        <SkillDetail onArchive={() => {}} onEdit={() => {}} skill={installedPdf} />
      </Providers>
    )

    expect(await screen.findByText('pdf')).toBeTruthy()
    expect(screen.getByText('PDF files: create, read, merge, fill, OCR, edit text.')).toBeTruthy()
  })
})
