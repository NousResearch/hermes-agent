/* Memory view object. In Python this is available as "memoryview". */

#ifndef Py_MEMORYOBJECT_H
#define Py_MEMORYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_LIMITED_API
PyAPI_DATA(PyTypeObject) _PyManagedBuffer_Type;
#endif
PyAPI_DATA(PyTypeObject) PyMemoryView_Type;

#define PyMemoryView_Check(op) Py_IS_TYPE(op, &PyMemoryView_Type)

#ifndef Py_LIMITED_API
/* Get a pointer to the memoryview's private copy of the exporter's buffer. */
#define PyMemoryView_GET_BUFFER(op) (&((PyMemoryViewObject *)(op))->view)
/* Get a pointer to the exporting object (this may be NULL!). */
#define PyMemoryView_GET_BASE(op) (((PyMemoryViewObject *)(op))->view.obj)
#endif

PyAPI_FUNC(PyObject *) PyMemoryView_FromObject(PyObject *base);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000
PyAPI_FUNC(PyObject *) PyMemoryView_FromMemory(char *mem, Py_ssize_t size,
                                               int flags);
#endif
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030b0000
PyAPI_FUNC(PyObject *) PyMemoryView_FromBuffer(const Py_buffer *info);
#endif
PyAPI_FUNC(PyObject *) PyMemoryView_GetContiguous(PyObject *base,
                                                  int buffertype,
                                                  char order);


/* The structs are declared here so that macros can work, but they shouldn't
   be considered public. Don't access their fields directly, use the macros
   and functions instead! */
#ifndef Py_LIMITED_API
#define _Py_MANAGED_BUFFER_RELEASED    0x001  /* access to exporter blocked */
#define _Py_MANAGED_BUFFER_FREE_FORMAT 0x002  /* free format */
typedef struct {
    PyObject_HEAD
    int flags;          /* state flags */
    Py_ssize_t exports; /* number of direct memoryview exports */
    Py_buffer master; /* snapshot buffer obtained from the original exporter */
} _PyManagedBufferObject;


/* memoryview state flags */
#define _Py_MEMORYVIEW_RELEASED    0x001  /* access to master buffer blocked */
#define _Py_MEMORYVIEW_C           0x002  /* C-contiguous layout */
#define _Py_MEMORYVIEW_FORTRAN     0x004  /* Fortran contiguous layout */
#define _Py_MEMORYVIEW_SCALAR      0x008  /* scalar: ndim = 0 */
#define _Py_MEMORYVIEW_PIL         0x010  /* PIL-style layout */

typedef struct {
    PyObject_VAR_HEAD
    _PyManagedBufferObject *mbuf; /* managed buffer */
    Py_hash_t hash;               /* hash value for read-only views */
    int flags;                    /* state flags */
    Py_ssize_t exports;           /* number of buffer re-exports */
    Py_buffer view;               /* private copy of the exporter's view */
    PyObject *weakreflist;
    Py_ssize_t ob_array[1];       /* shape, strides, suboffsets */
} PyMemoryViewObject;
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_MEMORYOBJECT_H */
