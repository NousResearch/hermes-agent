#ifndef Py_INTERNAL_CONTEXT_H
#define Py_INTERNAL_CONTEXT_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_hamt.h"   /* PyHamtObject */


extern PyTypeObject _PyContextTokenMissing_Type;

/* runtime lifecycle */

PyStatus _PyContext_Init(PyInterpreterState *);
void _PyContext_Fini(PyInterpreterState *);


/* other API */

#ifndef WITH_FREELISTS
// without freelists
#  define PyContext_MAXFREELIST 0
#endif

#ifndef PyContext_MAXFREELIST
#  define PyContext_MAXFREELIST 255
#endif

struct _Py_context_state {
#if PyContext_MAXFREELIST > 0
    // List of free PyContext objects
    PyContext *freelist;
    int numfree;
#endif
};

struct _pycontextobject {
    PyObject_HEAD
    PyContext *ctx_prev;
    PyHamtObject *ctx_vars;
    PyObject *ctx_weakreflist;
    int ctx_entered;
};


struct _pycontextvarobject {
    PyObject_HEAD
    PyObject *var_name;
    PyObject *var_default;
    PyObject *var_cached;
    uint64_t var_cached_tsid;
    uint64_t var_cached_tsver;
    Py_hash_t var_hash;
};


struct _pycontexttokenobject {
    PyObject_HEAD
    PyContext *tok_ctx;
    PyContextVar *tok_var;
    PyObject *tok_oldval;
    int tok_used;
};


#endif /* !Py_INTERNAL_CONTEXT_H */
