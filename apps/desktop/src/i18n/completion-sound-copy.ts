// Completion-sound settings copy lives together to keep the locale facades small.
export interface CompletionSoundCopy {
  completionSoundTitle: string
  completionSoundDesc: string
  completionSoundPreview: string
  completionSoundVolumeTitle: string
  completionSoundVolumeDesc: string
}

export const completionSoundCopy = {
  en: {
    completionSoundTitle: 'Completion Sound',
    completionSoundDesc: 'Plays when an agent turn finishes. Pick a preset and preview it here.',
    completionSoundPreview: 'Preview',
    completionSoundVolumeTitle: 'Completion Sound Volume',
    completionSoundVolumeDesc:
      'Scales the turn-end cue and its previews. Raise it when the chime is hard to hear on a TV or external speakers.'
  },
  ar: {
    completionSoundTitle: 'صوت الاكتمال',
    completionSoundDesc: 'يُشغّل عند انتهاء دور الوكيل. اختر إعدادا مسبقا وعاينه هنا.',
    completionSoundPreview: 'معاينة',
    completionSoundVolumeTitle: 'مستوى صوت الاكتمال',
    completionSoundVolumeDesc:
      'يضبط صوت نهاية الدور والمعاينة. ارفعه عندما يكون الجرس بالكاد يُسمع على تلفاز أو مكبرات صوت خارجية.'
  },
  de: {
    completionSoundTitle: 'Abschluss-Sound',
    completionSoundDesc:
      'Wird abgespielt, wenn ein Agent-Turn endet. Wählen Sie eine Vorlage aus und hören Sie sie hier an.',
    completionSoundPreview: 'Vorschau',
    completionSoundVolumeTitle: 'Lautstärke des Abschluss-Sounds',
    completionSoundVolumeDesc:
      'Skaliert den Ton am Rundenende und dessen Vorschau. Erhöhen, wenn der Klang über TV oder externe Lautsprecher kaum zu hören ist.'
  },
  es: {
    completionSoundTitle: 'Sonido de finalización',
    completionSoundDesc:
      'Reproduce un sonido cuando termina el turno de un agente. Elige un ajuste predefinido y pruébalo aquí.',
    completionSoundPreview: 'Vista previa',
    completionSoundVolumeTitle: 'Volumen del sonido de finalización',
    completionSoundVolumeDesc:
      'Ajusta el aviso de fin de turno y su vista previa. Súbelo si apenas se oye en un televisor o altavoces externos.'
  },
  fr: {
    completionSoundTitle: 'Son de fin',
    completionSoundDesc: "Se joue à la fin d'un tour d'agent. Choisissez un préréglage et prévisualisez-le ici.",
    completionSoundPreview: 'Aperçu',
    completionSoundVolumeTitle: 'Volume du son de fin',
    completionSoundVolumeDesc:
      "Ajuste le son de fin de tour et son aperçu. Augmentez-le si le carillon s'entend à peine sur un téléviseur ou des haut-parleurs externes."
  },
  ja: {
    completionSoundTitle: '完了サウンド',
    completionSoundDesc: 'エージェントのターン終了時に再生されます。プリセットを選んでここで試聴できます。',
    completionSoundPreview: '試聴',
    completionSoundVolumeTitle: '完了サウンドの音量',
    completionSoundVolumeDesc:
      'ターン終了時の通知音と試聴の音量を調整します。テレビや外部スピーカーで聞こえにくいときに上げてください。'
  },
  ru: {
    completionSoundTitle: 'Звук завершения',
    completionSoundDesc: 'Воспроизводится, когда ход агента завершён. Выберите пресет и прослушайте здесь.',
    completionSoundPreview: 'Прослушать',
    completionSoundVolumeTitle: 'Громкость звука завершения',
    completionSoundVolumeDesc:
      'Масштабирует сигнал конца хода и его прослушивание. Увеличьте, если звук плохо слышен на телевизоре или внешних колонках.'
  },
  zh: {
    completionSoundTitle: '完成提示音',
    completionSoundDesc: '智能体回合结束时播放。可在此选择预设并预览。',
    completionSoundPreview: '预览',
    completionSoundVolumeTitle: '完成提示音音量',
    completionSoundVolumeDesc: '调节回合结束提示音及试听的音量。在电视或外接音箱上听不清时可调高。'
  },
  'zh-hant': {
    completionSoundTitle: '完成提示音',
    completionSoundDesc: '代理回合結束時播放。可在此選擇預設並預覽。',
    completionSoundPreview: '預覽',
    completionSoundVolumeTitle: '完成提示音音量',
    completionSoundVolumeDesc: '調整回合結束提示音與試聽的音量。在電視或外接喇叭上聽不清楚時可調高。'
  }
} satisfies Record<string, CompletionSoundCopy>
