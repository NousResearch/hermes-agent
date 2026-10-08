import type { Translations } from './types'

export const ruQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'Пропустить настройку',
  back: 'Назад',
  trailLabel: 'Ваши ответы',
  changeAnswer: 'Изменить этот ответ',
  retry: 'Повторить',
  saveFailed: 'Не удалось отметить настройку как завершённую',
  skipped: 'Пропущено',
  otherLabel: 'Другой ответ',
  kinds: {
    accent: 'Цвет акцента',
    layout: 'Макет',
    local: 'Локальная модель',
    apps: 'Приложения на этом компьютере',
    connectors: 'Коннекторы'
  },
  name: {
    greeting: 'Привет, я Hermes.',
    question: 'Как к вам обращаться?',
    placeholder: 'Ваше имя'
  },
  accent: {
    title: 'Выберите цвет',
    custom: 'Свой цвет'
  },
  layout: {
    title: 'Как вы хотите работать?',
    basic: 'Базовый',
    basicDetail: 'Для общения с Hermes.',
    elite: 'Элитный',
    eliteDetail: 'Для разработчиков: терминал, файлы, диффы.'
  },
  local: {
    titleSpark: 'Ваш Spark может запускать Hermes офлайн',
    title: kind => `Этот ${kind} может запускать модель локально`,
    download: model => `Скачать ${model}`,
    downloadDetail: 'Загрузка пойдёт в фоне, когда начнётся ваш первый чат.',
    notNow: 'Не сейчас',
    notNowDetail: 'Останется в меню моделей как «Запустить локально».',
    chip: 'Чип',
    gpu: 'GPU',
    memory: 'Память',
    os: 'ОС',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'Использовать Hermes в этих приложениях?',
    plugin: 'Плагин',
    pluginNotRunning: 'Плагин · приложение не запущено',
    approve: 'Каждую установку вы подтвердите в первом чате.',
    found: 'Найдено на этом компьютере. Неустановленные приложения не показаны.'
  },
  connectors: {
    title: 'Подключить приложения?',
    checking: 'Проверяем, какие приложения может подключить ваш аккаунт…',
    offered: 'Это список, доступный вашему аккаунту сейчас. Вход в каждое приложение — в первом чате.'
  },
  task: {
    title: 'С чего начнём?',
    question: 'Выберите вариант или напишите свой.',
    blender: 'Собрать сцену в Blender',
    nvidia: 'Настроить мою GPU',
    broadcast: 'Настроить микрофон и камеру',
    brief: 'Сводка на сегодня',
    tidy: 'Разобрать папку «Загрузки»'
  },
  tour: {
    title: 'Показать, что где?',
    question: 'Быстрый обзор потом?',
    quick: 'Покажи',
    quickDetail: 'Пара остановок, меньше минуты.',
    none: 'Разберусь сам',
    noneDetail: 'Попросить Hermes об обзоре можно позже.'
  },
  start: {
    title: name => (name ? `Готово, ${name}.` : 'Готово.'),
    sub: 'Это первое сообщение, которое получит Hermes. Сначала просьба, остальное ниже.',
    aboutMe: 'Обо мне:',
    beforeYouStart: 'Перед началом:',
    noTask: 'Задача не выбрана: Hermes спросит, что вы хотите сделать.',
    action: 'Начать',
    waiting: 'Готовим ваш бесплатный аккаунт…',
    failed: 'Не удалось начать первый чат'
  },
  summary: {
    custom: 'Свой',
    local: model => `Локально: ${model}`,
    localNo: 'Локально: не сейчас',
    noApps: 'Без приложений',
    noConnectors: 'Без коннекторов',
    more: (label, count) => `${label} +${count}`,
    tour: 'Обзор',
    noTour: 'Без обзора'
  },
  status: {
    settingUp: 'Настраиваем ваш бесплатный аккаунт',
    stillSettingUp: 'Всё ещё настраиваем ваш бесплатный аккаунт',
    unavailable: 'Бесплатный аккаунт недоступен',
    ready: 'Nous · бесплатный тариф',
    downloadingEngine: 'Загрузка локального движка',
    downloading: model => `Загрузка ${model}`
  },
  settings: {
    title: 'Пройти настройку заново',
    description: 'Снова открывает вопросы первого запуска. Ответы применяются к профилю по умолчанию.',
    action: 'Начать настройку',
    failed: 'Не удалось начать настройку'
  }
}
