"use client";

import dynamic from 'next/dynamic';
import { useI18n } from 'fumadocs-ui/contexts/i18n';
import { useTheme } from 'fumadocs-ui/provider/base';
import type { SharedProps } from 'fumadocs-ui/contexts/search';
import config from '../../public/search-config.json';
import { searchFacetFilters, transformResultUrl } from '@/lib/search';

const SearchModal = dynamic(() => import('@docsearch/react').then((module) => module.DocSearchModal), { ssr: false });

export function DocSearchDialog({ open, onOpenChange }: SharedProps) {
  const { locale = 'en' } = useI18n();
  const { resolvedTheme } = useTheme();
  if (!open) return null;
  return (
    <SearchModal
      theme={resolvedTheme === 'dark' ? 'dark' : 'light'}
      appId={config.appId}
      apiKey={config.apiKey}
      indices={[{ name: config.indexName, searchParameters: { facetFilters: searchFacetFilters(locale) } }]}
      initialScrollY={window.scrollY}
      onClose={() => onOpenChange(false)}
      placeholder={locale === 'zh-Hans' ? '搜索文档' : 'Search documentation'}
      insights={false}
      translations={locale === 'zh-Hans' ? {
        searchBox: { clearButtonTitle: '清除查询', clearButtonAriaLabel: '清除查询', closeButtonText: '关闭', closeButtonAriaLabel: '关闭搜索', searchInputLabel: '搜索' },
        startScreen: { recentSearchesTitle: '最近搜索', noRecentSearchesText: '没有最近搜索', saveRecentSearchButtonTitle: '收藏此搜索', removeRecentSearchButtonTitle: '移除此搜索', favoriteSearchesTitle: '收藏', removeFavoriteSearchButtonTitle: '移除收藏' },
        footer: { selectText: '选择', selectKeyAriaLabel: '回车键', navigateText: '切换', navigateUpKeyAriaLabel: '向上箭头', navigateDownKeyAriaLabel: '向下箭头', closeText: '关闭', closeKeyAriaLabel: 'Esc 键', poweredByText: '搜索服务由' },
        errorScreen: { titleText: '无法获取结果', helpText: '请检查您的网络连接。' },
        noResultsScreen: { noResultsText: '没有找到相关结果', suggestedQueryText: '请尝试搜索', reportMissingResultsText: '找不到您想要的内容？', reportMissingResultsLinkText: '告诉我们。' },
        resultsScreen: { resultsSectionTitle: '文档' },
      } : undefined}
      transformItems={(items) => items.flatMap((item) => {
        const url = transformResultUrl(item.url);
        return url === null ? [] : [{ ...item, url }];
      })}
    />
  );
}
