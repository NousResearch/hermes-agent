
#ifndef Py_CURSES_H
#define Py_CURSES_H

#ifdef __APPLE__
/*
** On Mac OS X 10.2 [n]curses.h and stdlib.h use different guards
** against multiple definition of wchar_t.
*/
#ifdef _BSD_WCHAR_T_DEFINED_
#define _WCHAR_T
#endif
#endif /* __APPLE__ */

/* On FreeBSD, [n]curses.h and stdlib.h/wchar.h use different guards
   against multiple definition of wchar_t and wint_t. */
#if defined(__FreeBSD__) && defined(_XOPEN_SOURCE_EXTENDED)
# ifndef __wchar_t
#   define __wchar_t
# endif
# ifndef __wint_t
#   define __wint_t
# endif
#endif

#if !defined(HAVE_CURSES_IS_PAD) && defined(WINDOW_HAS_FLAGS)
/* The following definition is necessary for ncurses 5.7; without it,
   some of [n]curses.h set NCURSES_OPAQUE to 1, and then Python
   can't get at the WINDOW flags field. */
#define NCURSES_OPAQUE 0
#endif

#ifdef HAVE_NCURSES_H
#include <ncurses.h>
#else
#include <curses.h>
#endif

#ifdef HAVE_NCURSES_H
/* configure was checking <curses.h>, but we will
   use <ncurses.h>, which has some or all these features. */
#if !defined(WINDOW_HAS_FLAGS) && !(NCURSES_OPAQUE+0)
#define WINDOW_HAS_FLAGS 1
#endif
#if !defined(HAVE_CURSES_IS_PAD) && NCURSES_VERSION_PATCH+0 >= 20090906
#define HAVE_CURSES_IS_PAD 1
#endif
#ifndef MVWDELCH_IS_EXPRESSION
#define MVWDELCH_IS_EXPRESSION 1
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PyCurses_API_pointers 4

/* Type declarations */

typedef struct {
    PyObject_HEAD
    WINDOW *win;
    char *encoding;
} PyCursesWindowObject;

#define PyCursesWindow_Check(v) Py_IS_TYPE(v, &PyCursesWindow_Type)

#define PyCurses_CAPSULE_NAME "_curses._C_API"


#ifdef CURSES_MODULE
/* This section is used when compiling _cursesmodule.c */

#else
/* This section is used in modules that use the _cursesmodule API */

static void **PyCurses_API;

#define PyCursesWindow_Type (*_PyType_CAST(PyCurses_API[0]))
#define PyCursesSetupTermCalled  {if (! ((int (*)(void))PyCurses_API[1]) () ) return NULL;}
#define PyCursesInitialised      {if (! ((int (*)(void))PyCurses_API[2]) () ) return NULL;}
#define PyCursesInitialisedColor {if (! ((int (*)(void))PyCurses_API[3]) () ) return NULL;}

#define import_curses() \
    PyCurses_API = (void **)PyCapsule_Import(PyCurses_CAPSULE_NAME, 1);

#endif

/* general error messages */
static const char catchall_ERR[]  = "curses function returned ERR";
static const char catchall_NULL[] = "curses function returned NULL";

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_CURSES_H) */

