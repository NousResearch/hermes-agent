#ifndef Py_INTERNAL_LIST_H
#define Py_INTERNAL_LIST_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "listobject.h"           // _PyList_CAST()


/* runtime lifecycle */

extern void _PyList_Fini(PyInterpreterState *);


/* other API */

#ifndef WITH_FREELISTS
// without freelists
#  define PyList_MAXFREELIST 0
#endif

/* Empty list reuse scheme to save calls to malloc and free */
#ifndef PyList_MAXFREELIST
#  define PyList_MAXFREELIST 80
#endif

struct _Py_list_state {
#if PyList_MAXFREELIST > 0
    PyListObject *free_list[PyList_MAXFREELIST];
    int numfree;
#endif
};

#define _PyList_ITEMS(op) (_PyList_CAST(op)->ob_item)

extern int
_PyList_AppendTakeRefListResize(PyListObject *self, PyObject *newitem);

static inline int
_PyList_AppendTakeRef(PyListObject *self, PyObject *newitem)
{
    assert(self != NULL && newitem != NULL);
    assert(PyList_Check(self));
    Py_ssize_t len = PyList_GET_SIZE(self);
    Py_ssize_t allocated = self->allocated;
    assert((size_t)len + 1 < PY_SSIZE_T_MAX);
    if (allocated > len) {
        PyList_SET_ITEM(self, len, newitem);
        Py_SET_SIZE(self, len + 1);
        return 0;
    }
    return _PyList_AppendTakeRefListResize(self, newitem);
}

#ifdef __cplusplus
}
#endif
#endif   /* !Py_INTERNAL_LIST_H */
