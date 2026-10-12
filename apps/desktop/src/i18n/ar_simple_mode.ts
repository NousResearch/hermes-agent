import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into ar.ts.
export const arSimpleMode = {
  interfaceMode: {
    title: 'وضع الواجهة',
    hint: 'يغيّر ما يظهر، وليس ما يستطيع Hermes فعله.',
    sessionNote: 'يحدده الوضع البسيط. التغيير هنا يستمر لهذه الجلسة فقط؛ بدّل إلى المتقدم لجعله خيارك.',
    switchToAdvanced: 'التبديل إلى المتقدم',
    simple: {
      label: 'بسيط',
      description: 'للتحدث مع Hermes. الشريط الجانبي والدردشة؛ بلا طرفية أو لوحات ملفات أو فروقات.'
    },
    advanced: {
      label: 'متقدم',
      description: 'للمطورين. الطرفية والملفات والفروقات وشريط الحالة والتخطيطات، كما أعددتها.'
    }
  },
  chatStarters: {
    label: 'أفكار للبدء',
    intro: 'اسألني أي شيء — أو ابدأ بإحدى هذه الأفكار.',
    email: {
      title: 'ساعدني في كتابة رسالة بريد إلكتروني',
      prompt: 'ساعدني في كتابة رسالة بريد إلكتروني. إليك لمن سأرسلها وما أريد قوله: '
    },
    dinners: {
      title: 'خطّط لوجبات العشاء هذا الأسبوع',
      prompt: 'خطّط لخمس وجبات عشاء سهلة لهذا الأسبوع، وأعطني قائمة تسوّق واحدة لها جميعًا.'
    },
    summarize: {
      title: 'لخّص مستندًا',
      prompt: 'سأرفق ملفًا — أعطني أهم النقاط فيه.'
    },
    lookup: {
      title: 'ابحث لي عن شيء',
      prompt: 'ابحث لي عن هذا وأخبرني بما تجده: '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
