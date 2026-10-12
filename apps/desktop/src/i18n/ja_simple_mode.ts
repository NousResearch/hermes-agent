import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into ja.ts.
export const jaSimpleMode = {
  interfaceMode: {
    title: 'インターフェースモード',
    hint: '表示される内容が変わるだけで、Hermes にできることは変わりません。',
    sessionNote:
      'シンプルモードで設定されています。ここでの変更はこのセッション中のみ有効です。自分の設定にするには詳細モードに切り替えてください。',
    switchToAdvanced: '詳細モードに切り替え',
    simple: {
      label: 'シンプル',
      description:
        'Hermes と話すための表示。サイドバーとチャットのみ。ターミナル、ファイル、差分のペインは表示しません。'
    },
    advanced: {
      label: '詳細',
      description: '開発者向け。ターミナル、ファイル、差分、ステータスバー、レイアウトを設定したとおりに。'
    }
  },
  chatStarters: {
    label: 'こんなことから始めてみましょう',
    intro: '何でも聞いてください。または、ここから始めてみましょう。',
    email: {
      title: 'メールを書くのを手伝って',
      prompt: 'メールを書くのを手伝ってください。宛先と伝えたい内容は次のとおりです: '
    },
    dinners: {
      title: '今週の夕食を考えて',
      prompt: '今週の簡単な夕食を 5 つ考えて、まとめて使える買い物リストを 1 つ作ってください。'
    },
    summarize: {
      title: '文書を要約して',
      prompt: 'ファイルを添付します。要点を教えてください。'
    },
    lookup: {
      title: '調べものをお願い',
      prompt: 'これについて調べて、わかったことを教えてください: '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
